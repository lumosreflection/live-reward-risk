
import streamlit as st
import pandas as pd
import matplotlib

matplotlib.use('Agg')
# ========== 修复matplotlib中文方框【新增这一段】 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示
import matplotlib.pyplot as plt
from user_risk_func import (
    detect_user_risk,
    build_user_feature,
    train_lgb_model,
    calc_risk_flags,
    get_risk_level,
    speech_to_text,
    export_insurance_evidence_full,
    face_verify_sdk
)



def calc_risk_flags(raw_df, user_id, large_threshold):
    print(f"【调试】收到的大额阈值large_threshold = {large_threshold}")
    # 后面原有代码不变
    
    ud = raw_df[raw_df["user_id"] == user_id].copy()
    ud["gift_time"] = pd.to_datetime(ud["gift_time"])
    ud = ud.sort_values("gift_time")
    flags = {}
    
    # 1. risk_fast_charge：1小时内至少3笔 → 短时间高频充值
    ud["time_diff_h"] = ud["gift_time"].diff().dt.total_seconds() / 3600
    fast_count = ((ud["time_diff_h"] < 1) & (ud["time_diff_h"].notna())).sum()
    flags["risk_fast_charge"] = True if fast_count >= 2 else False
    
    # 2. risk_single_large：存在单笔超过大额阈值
    flags["risk_single_large"] = (ud["gift_value"] > large_threshold).any()
    
    # 3. risk_spike：消费突然暴涨，最后一笔 > 前面所有笔均值的2倍（至少2条记录才判断）
    if len(ud) > 1:
        history_mean = ud["gift_value"].iloc[:-1].mean()
        last_val = ud["gift_value"].iloc[-1]
        flags["risk_spike"] = True if last_val > 2 * history_mean else False
    else:
        flags["risk_spike"] = False
    
    # 4. risk_silent_burst：第一笔和第二笔间隔>14天，沉寂后集中充值
    if len(ud) >= 2:
        gap_day = (ud["gift_time"].iloc[1] - ud["gift_time"].iloc[0]).total_seconds() / (3600 * 24)
        flags["risk_silent_burst"] = True if gap_day > 14 else False
    else:
        flags["risk_silent_burst"] = False
    
    # 5. risk_night_large：存在夜间大额打赏
    ud["hour"] = ud["gift_time"].dt.hour
    ud["is_night"] = ud["hour"].apply(lambda h: 1 if h >= 23 or h <= 5 else 0)
    ud["is_large"] = ud["gift_value"] > large_threshold
    flags["risk_night_large"] = ((ud["is_night"] == 1) & (ud["is_large"])).any()
    
    flags["user_id"] = user_id
    return flags


# 上面都是细分预警给家长和三方看的

#st.set_page_config(page_title="用户行为风控", layout="wide")
st.title("直播用户时序行为风险检测")#KuaiLive特征框架+夜间行为增强

# 初始化会话状态
if "df_data" not in st.session_state:
    st.session_state.df_data = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

uploaded_file = st.file_uploader("上传送礼数据csv（gift.csv格式）", type="csv")
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("文件上传成功")
    st.session_state.df_data = df
elif st.button("使用本地仿真gift.csv"):
    try:
        df_local = pd.read_csv("gift.csv")
        st.session_state.df_data = df_local
        st.success("加载本地仿真数据集")
        st.write(f"读取到总行数：{df_local.shape[0]}，总列数：{df_local.shape[1]}")
        st.write(f"数据集内独立用户数量：{df_local['user_id'].nunique()}")
    except Exception as e:
        st.error(f"读取本地gift.csv失败：{e}")

df = st.session_state.df_data

if df is not None:
    st.subheader("原始送礼数据预览")
    st.dataframe(df, width="stretch")
    
    # ========== 新增：网页调参滑块 ==========
    st.subheader("模型参数设置")
    contamination = st.slider("异常样本比例 contamination", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
    large_threshold = st.slider("单笔大额打赏阈值", min_value=50, max_value=500, value=200, step=10)
    
    # =========粘贴下拉框代码=========
    model_choice = st.selectbox(
        "选择风险检测模型",
        options=["IsolationForest孤立森林", "LightGBM伪标签训练三级预警"],
        index=0
    )
    # ===============================
    
    # 运行检测按钮
    if st.button("运行用户行为异常检测"):
        try:
            # 根据下拉框选择分支执行
            if model_choice == "IsolationForest孤立森林":
                # 进度提示弹窗
                with st.spinner("正在计算用户行为特征，执行PyOD异常检测，请稍候..."):
                    result_df = detect_user_risk(df, contamination=contamination, large_threshold=large_threshold)
                    # ========== 批量计算所有用户风险预警标签 ==========
                    risk_flag_list = []
                    for uid in result_df["user_id"]:
                        one_flag = calc_risk_flags(df, uid, large_threshold)
                        risk_flag_list.append(one_flag)
                    risk_df = pd.DataFrame(risk_flag_list)
                    # =========布尔值转文字，解决方框问题========
                    bool_cols = ["risk_fast_charge", "risk_single_large", "risk_spike", "risk_silent_burst",
                                 "risk_night_large"]
                    for col in bool_cols:
                        risk_df[col] = risk_df[col].map({True: "存在风险", False: "无异常"})
                    # 合并风险标签到模型结果表
                    result_df = pd.merge(result_df, risk_df, on="user_id", how="left")
                    # 存入session
                    st.session_state.result_df = result_df.copy()
                
                # spinner结束自动消失，展示结果
                st.success("风险检测任务执行完成！")
                st.subheader("全部用户风险结果表")
                st.dataframe(result_df, width="stretch")
                high_risk = result_df[result_df["is_high_risk"] == 1]
                st.subheader("高风险用户名单")
                st.dataframe(high_risk)
                st.write(f"高风险用户数量：{len(high_risk)}")
                # 散点图
                fig, ax = plt.subplots(figsize=(10, 6))
                normal = result_df[result_df["is_high_risk"] == 0]
                risk = result_df[result_df["is_high_risk"] == 1]
                ax.scatter(normal["total_gift_money"], normal["total_gift_cnt"], c="#4ECDC4", label="正常用户", alpha=0.7)
                ax.scatter(risk["total_gift_money"], risk["total_gift_cnt"], c="#FF6B6B", label="高风险用户", s=80)
                ax.set_xlabel("用户总打赏金额")
                ax.set_ylabel("用户总送礼次数")
                ax.set_title("用户风险分布散点图")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                # 下载按钮
                csv_out = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载全部用户风险结果csv",
                    data=csv_out,
                    file_name="user_risk_result.csv",
                    mime="text/csv"
                )
            
            else:
                # ============LightGBM分支==========
                with st.spinner("LightGBM模型训练中，正在计算三级风险预警..."):
                    user_feat, feat_cols = build_user_feature(df, contamination, large_threshold)
                    user_feat, lgb_model, feat_import = train_lgb_model(user_feat, feat_cols)
                    
                    # 批量计算业务风险标记
                    flag_list = []
                    for uid in user_feat["user_id"]:
                        f = calc_risk_flags(df, uid, large_threshold)
                        flag_list.append(f)
                    flag_df = pd.DataFrame(flag_list)
                    user_feat = pd.merge(user_feat, flag_df, on="user_id", how="left")
                    # 生成三级预警
                    user_feat["risk_level"] = user_feat.apply(get_risk_level, axis=1)
                st.success("模型计算完成！")
                st.subheader("用户风险结果（LightGBM与三级预警）")
                st.dataframe(user_feat[["user_id", "total_gift_money", "lgb_risk_prob", "lgb_risk_pred", "risk_level"]])
                # 展示特征重要性图
                st.subheader("LightGBM特征重要性")
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(feat_import["feature"], feat_import["importance"])
                ax.invert_yaxis()
                ax.set_xlabel("特征重要性")
                ax.set_title("LightGBM 特征重要性")
                plt.tight_layout()
                st.pyplot(fig, width='stretch')
                # 展示AUC指标
                from sklearn.metrics import roc_auc_score
                
                auc = roc_auc_score(user_feat["is_high_risk"], user_feat["lgb_risk_prob"])
                st.metric(label="LightGBM AUC", value=f"{auc:.4f}")
        except Exception as e:
            st.error(f"风险检测运行失败：{e}")
    
   
    
    # ===================== 【时序模块移到按钮外面！！】 =====================
    result_df = st.session_state.result_df
    if result_df is not None:
        st.divider()
        st.subheader("单用户打赏行为与时序查看")
        user_list = sorted(df["user_id"].unique())
        # 会话状态记录选中用户，默认取列表第一个
        if "selected_uid" not in st.session_state or st.session_state.selected_uid not in user_list:
            st.session_state.selected_uid = user_list[0]
        
        selected_user = st.selectbox(
            "选择用户ID",
            user_list,
            index=user_list.index(st.session_state.selected_uid),
            key="user_selector_main"
        )
        # 更新选中id到session
        st.session_state.selected_uid = selected_user
        
        user_data = df[df["user_id"] == selected_user].copy()
        # =========新增判断：该用户没有任何打赏记录=========
        if len(user_data) == 0:
            st.warning("该用户没有任何打赏记录！")
        else:
            user_data["gift_time"] = pd.to_datetime(user_data["gift_time"])
            user_data = user_data.sort_values("gift_time")
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(user_data["gift_time"], user_data["gift_value"], marker="o", linestyle="-", color="#2E86AB")
            ax2.scatter(user_data["gift_time"], user_data["gift_value"], c="#A23B72", s=40)
            ax2.set_xlabel("打赏时间")
            ax2.set_ylabel("单笔打赏金额")
            ax2.set_title(f"用户 {selected_user} 打赏时序曲线")
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=30)
            st.pyplot(fig2)
            
            # 展示选中用户的风险预警详情
            st.subheader(f"用户{selected_user}风险预警详情")
            user_risk_info = result_df[result_df["user_id"] == selected_user][
                ["risk_fast_charge", "risk_single_large", "risk_spike", "risk_silent_burst", "risk_night_large"]
            ]
            st.dataframe(user_risk_info)
            
            st.write(f"该用户总打赏金额：{user_data['gift_value'].sum()}")
            st.write(f"该用户总打赏笔数：{len(user_data)}")

            # ========== 查看该用户风险档案（跳转风险档案页） ==========
            if st.button("查看该用户风险档案", key="btn_goto_archive"):
                st.session_state["guard_jump_user"] = selected_user
                st.switch_page("pages/6_风险档案.py")
            st.caption("跳转到风险档案页查看该账号全量风险记录与投保状态")

            # ========== 模拟充值 + 人脸核验（嵌入业务流程，绑定当前选中用户） ==========
            st.markdown("**模拟充值与身份核验**")
            cur_row = result_df[result_df["user_id"] == selected_user].iloc[0]
            recharge_amt = st.number_input("模拟充值金额（元）", min_value=0, value=500,
                                           step=50, key="face_recharge_amt")

            if "face_verify_log" not in st.session_state:
                st.session_state.face_verify_log = []
            if "face_target_user" not in st.session_state:
                st.session_state.face_target_user = None

            if st.button("模拟该账号发起充值", key="btn_start_recharge"):
                if cur_row["is_high_risk"] == 1:
                    # 高风险：触发人脸核验
                    st.session_state.face_target_user = selected_user
                else:
                    # 正常账号：直接放行
                    st.session_state.face_target_user = None
                    st.session_state.face_verify_log.append({
                        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": selected_user,
                        "recharge_amount": recharge_amt,
                        "verify_required": "否",
                        "verify_result": "无需核验",
                        "final_action": "充值放行"
                    })
                    st.success("账号风险正常，充值直接放行")

            # 人脸核验弹窗（仅对当前选中且被标记的用户显示）
            if st.session_state.face_target_user == selected_user:
                with st.container(border=True):
                    st.warning("检测到高风险账号发起充值，触发强制人脸核验")
                    st.markdown("请完成本人人脸核验")
                    st.image("https://via.placeholder.com/320x240?text=face+verify", width=320)
                    st.caption("实际业务调用平台人脸SDK；本Demo仅UI模拟，不执行真实比对")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        if st.button("核验通过", key="btn_face_pass"):
                            sdk = face_verify_sdk(selected_user, simulate=True, pass_rate=1.0)
                            st.session_state.face_verify_log.append({
                                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "user_id": selected_user,
                                "recharge_amount": recharge_amt,
                                "verify_required": "是",
                                "verify_result": "核验通过",
                                "verify_sdk": sdk["sdk_name"],
                                "confidence": sdk["confidence"],
                                "final_action": "充值放行"
                            })
                            st.session_state.face_target_user = None
                            st.success("人脸核验通过，充值继续")
                            st.rerun()
                    with fc2:
                        if st.button("核验不通过", key="btn_face_fail"):
                            sdk = face_verify_sdk(selected_user, simulate=True, pass_rate=0.0)
                            st.session_state.face_verify_log.append({
                                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "user_id": selected_user,
                                "recharge_amount": recharge_amt,
                                "verify_required": "是",
                                "verify_result": "核验不通过",
                                "verify_sdk": sdk["sdk_name"],
                                "confidence": sdk["confidence"],
                                "final_action": "拦截充值"
                            })
                            st.session_state.face_target_user = None
                            st.error("人脸核验未通过，拦截本次充值")
                            st.rerun()

            # 该账号的历史核验记录
            user_face_logs = [x for x in st.session_state.face_verify_log
                              if x["user_id"] == selected_user]
            if len(user_face_logs) > 0:
                st.markdown("**该账号核验记录**")
                st.dataframe(pd.DataFrame(user_face_logs), width="stretch", hide_index=True)

st.divider()
with st.expander("证据采集与分析工具箱（话术识别 / 证据包导出）", expanded=False):
    st.subheader("音频｜Whisper直播话术诱导风险识别")
    use_mock_asr = st.toggle("Mock模拟转写（关闭调用真实Whisper-base）", value=True)
    upload_audio = st.file_uploader("上传直播录音(wav/mp3)", type=["mp3","wav"])

    # 话术识别引擎选择
    engine_choice = st.radio(
        "话术识别引擎",
        options=["BERT微调模型（推荐）", "关键词规则"],
        horizontal=True
    )
    bert_threshold = st.slider("BERT判定阈值（概率≥阈值判为诱导）", 0.3, 0.7, 0.5, 0.05)

    # 初始化session存储话术风险结果
    if "speech_detect_result" not in st.session_state:
        st.session_state["speech_detect_result"] = None

    audio_text = ""
    if upload_audio is not None:
        # 第一步：转写
        with st.spinner("音频转写中..."):
            audio_text = speech_to_text(upload_audio, use_mock=use_mock_asr)
        st.success("音频转写完成")
        st.text_area("直播转写文本", value=audio_text, height=180)
        # 新音频转写后清空旧识别结果，避免显示上一次结果
        st.session_state["speech_detect_result"] = None

        # 第二步：按所选引擎做话术识别
        if st.button("运行直播话术风险识别"):
            with st.spinner("话术风险识别中（首次加载BERT约需10-30秒）..."):
                if engine_choice == "BERT微调模型":
                    from user_risk_func import long_text_speech_risk_detect_bert
                    speech_result = long_text_speech_risk_detect_bert(audio_text, threshold=bert_threshold)
                else:
                    from user_risk_func import long_text_speech_risk_detect_rule
                    speech_result = long_text_speech_risk_detect_rule(audio_text)
                st.session_state["speech_detect_result"] = speech_result

        res = st.session_state["speech_detect_result"]
        if res is not None:
            if res["whole_risk"]:
                st.error(f"本场直播存在诱导打赏风险，共识别到{len(res['risk_sentences'])}条风险语句")
                st.dataframe(pd.DataFrame(res["risk_sentences"]), use_container_width=True)
            else:
                st.success("本场直播话术未检测到诱导风险")
            with st.expander("查看全部分句识别结果"):
                st.dataframe(pd.DataFrame(res["all_sent_result"]), use_container_width=True)
    st.session_state["audio_text"] = audio_text


    st.divider()
    st.subheader("生成完整保险理赔证据包zip")
    st.info("证据包包含：用户风险总表、行为时间线；如有弹幕/音频会一并打包")

    if st.button("生成证据压缩包"):
        if "result_df" not in st.session_state or st.session_state.result_df is None:
            st.warning("请先运行【用户行为异常检测】生成风险结果！")
        elif "df_data" not in st.session_state or st.session_state.df_data is None:
            st.warning("缺少原始送礼数据集")
        else:
            with st.spinner("正在生成证据包……"):
                # 如果已经跑过弹幕模块，session可以存danmu_result，这里传进去
                danmu_input = st.session_state.get("danmu_result", None)
                # 改成：弹幕彻底不用，传None
                danmu_input = None
                # 从session拿转写文本
                speech_input_text = st.session_state.get("audio_text", "")
                zip_file_path = export_insurance_evidence_full(
                    gift_risk_df=st.session_state.result_df,
                    raw_gift_df=st.session_state.df_data,
                    danmu_df=danmu_input,
                    speech_text=speech_input_text
                )
            st.success("证据包生成完成！")
            with open(zip_file_path, "rb") as f:
                st.download_button(
                    label="📥下载保险证据包 zip",
                    data=f,
                    file_name="insurance_evidence_package.zip",
                    mime="application/zip"
                )



# 人脸核验已迁移至上方「单个用户打赏时序查看」模块：跟随选中用户触发，并记录核验结果
