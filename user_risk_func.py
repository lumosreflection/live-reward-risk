# user_risk_func.py
#用户风险函数，方便之后streamlit调用
import pandas as pd
from pyod.models.iforest import IsolationForest
import streamlit as st

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def setup_cn_font():
    """
    注册项目自带开源中文字体（Noto Sans CJK SC），
    解决云端Linux无中文字体导致matplotlib图表中文显示方框的问题
    """
    import os
    import matplotlib
    from matplotlib import font_manager
    _root = os.path.dirname(os.path.abspath(__file__))
    _font_file = os.path.join(_root, "fonts", "NotoSansCJKsc-Regular.otf")
    if os.path.exists(_font_file):
        font_manager.fontManager.addfont(_font_file)
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

def detect_user_risk(df_raw,contamination=0.08, large_threshold=200):
    #显示中间状态，方便修正bug
    # df_raw 是读进来的gift.csv原始数据
    df_raw["gift_time"] = pd.to_datetime(df_raw["gift_time"])
    # 按用户排序，计算送礼间隔
    df_raw = df_raw.sort_values(["user_id", "gift_time"])
    
    
    
    df_raw["time_diff_min"] = df_raw.groupby("user_id")["gift_time"].diff().dt.total_seconds() / 60

    # ========== 新增夜间行为特征 ==========
    # 提取小时，定义夜间：23:00 ~ 05:59
    df_raw["hour"] = df_raw["gift_time"].dt.hour
    df_raw["is_night"] = df_raw["hour"].apply(lambda h: 1 if h >= 23 or h <= 5 else 0)
    # 大额阈值从网页传入，不再硬编码
    #large_threshold = 200
    df_raw["is_large"] = (df_raw["gift_value"] > large_threshold).astype(int)
    df_raw["night_large"] = ((df_raw["is_night"] == 1) & (df_raw["is_large"] == 1)).astype(int)
    df_raw["night_gift_value"] = df_raw.apply(lambda row: row["gift_value"] if row["is_night"] == 1 else 0, axis=1)

    # 构造用户级特征（参考KuaiLive特征）
    user_feat = df_raw.groupby("user_id").agg(
        total_gift_money=("gift_value", "sum"),
        total_gift_cnt=("gift_value", "count"),
        room_count=("room_id", "nunique"),
        min_interval=("time_diff_min", "min"),
    # 新增夜间相关特征
        
        night_gift_money = ("night_gift_value", "sum"),
        night_gift_cnt = ("is_night", "sum"),
        night_large_cnt = ("night_large", "sum")
    ).reset_index()
    
    # 衍生特征：夜间金额占总打赏金额比例，防止除以0报错
    user_feat["night_ratio"] = user_feat.apply(
        lambda row: row["night_gift_money"] / row["total_gift_money"] if row["total_gift_money"] > 0 else 0,
        axis=1
    )
    # ==========新增调试打印==========
    print(f"【调试】收到的大额阈值large_threshold = {large_threshold}")
    print("====调试：用户特征前5行，观察night_large_cnt=====")
    print(user_feat[["user_id", "night_large_cnt"]].head())
    # 缺失值填充：第一次送礼没有间隔，填充一个很大的数
    user_feat["min_interval"] = user_feat["min_interval"].fillna(0)
    
    
    
    # 孤立森林异常检测（PyOD）
    feature_cols = [
        "total_gift_money",
        "total_gift_cnt",
        "room_count",
        "min_interval",
        "night_gift_money",
        "night_gift_cnt",
        "night_ratio",
        "night_large_cnt",
    ]
    
    #这里的0.08是触发异常的临界值，可以微调
    clf = IsolationForest(contamination=contamination, random_state=42)
    user_feat["anomaly_score"] = clf.fit_predict(user_feat[feature_cols])
    user_feat["risk_score"] = clf.decision_function(user_feat[feature_cols])
    
    # 标签转换：-1=异常(高风险)，1=正常
    user_feat["is_high_risk"] = user_feat["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)
    
    return user_feat

#LightGBM+三级风险
def build_user_feature(df_raw, contamination=0.08, large_threshold=200):
    df_raw["gift_time"] = pd.to_datetime(df_raw["gift_time"])
    df_raw = df_raw.sort_values(["user_id", "gift_time"])
    df_raw["time_diff_min"] = df_raw.groupby("user_id")["gift_time"].diff().dt.total_seconds() / 60

    # 夜间标记
    df_raw["hour"] = df_raw["gift_time"].dt.hour
    df_raw["is_night"] = df_raw["hour"].apply(lambda h: 1 if h >= 23 or h <=5 else 0)
    df_raw["is_large"] = (df_raw["gift_value"] > large_threshold).astype(int)
    df_raw["night_large"] = ((df_raw["is_night"] ==1) & (df_raw["is_large"] ==1)).astype(int)
    df_raw["night_gift_value"] = df_raw.apply(lambda row: row["gift_value"] if row["is_night"]==1 else 0, axis=1)

    # 用户级聚合特征
    user_feat = df_raw.groupby("user_id").agg(
        total_gift_money=("gift_value", "sum"),
        total_gift_cnt=("gift_value", "count"),
        room_count=("room_id", "nunique"),
        min_interval=("time_diff_min", "min"),
        night_gift_money = ("night_gift_value", "sum"),
        night_gift_cnt = ("is_night", "sum"),
        night_large_cnt = ("night_large", "sum")
    ).reset_index()

    user_feat["night_ratio"] = user_feat.apply(
        lambda row: row["night_gift_money"] / row["total_gift_money"] if row["total_gift_money"]>0 else 0, axis=1
    )
    user_feat["min_interval"] = user_feat["min_interval"].fillna(0)

    feature_cols = [
        "total_gift_money",
        "total_gift_cnt",
        "room_count",
        "min_interval",
        "night_gift_money",
        "night_gift_cnt",
        "night_ratio",
        "night_large_cnt",
    ]
    # 孤立森林生成伪标签
    clf_if = IsolationForest(contamination=contamination, random_state=42)
    user_feat["anomaly_score"] = clf_if.fit_predict(user_feat[feature_cols])
    user_feat["is_high_risk"] = user_feat["anomaly_score"].apply(lambda x:1 if x == -1 else 0)
    return user_feat, feature_cols

def train_lgb_model(user_feature_df, feature_cols):
    X = user_feature_df[feature_cols]
    y = user_feature_df["is_high_risk"]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    # 预测高风险概率 0~1
    user_feature_df["lgb_risk_prob"] = model.predict_proba(X)[:,1]
    user_feature_df["lgb_risk_pred"] = model.predict(X)

    # 特征重要性
    feat_import_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    auc = roc_auc_score(y, user_feature_df["lgb_risk_prob"])
    return user_feature_df, model, feat_import_df

def calc_risk_flags(raw_df, user_id, large_threshold):
    ud = raw_df[raw_df["user_id"] == user_id].copy()
    ud["gift_time"] = pd.to_datetime(ud["gift_time"])
    ud = ud.sort_values("gift_time")
    flags = {}

    ud["time_diff_h"] = ud["gift_time"].diff().dt.total_seconds() / 3600
    fast_count = ((ud["time_diff_h"] <1) & (ud["time_diff_h"].notna())).sum()
    flags["risk_fast_charge"] = True if fast_count >= 2 else False
    flags["risk_single_large"] = (ud["gift_value"]>large_threshold).any()

    if len(ud) > 1:
        history_mean = ud["gift_value"].iloc[:-1].mean()
        last_val = ud["gift_value"].iloc[-1]
        flags["risk_spike"] = True if last_val > 2 * history_mean else False
    else:
        flags["risk_spike"] = False

    if len(ud)>=2:
        gap_day = (ud["gift_time"].iloc[1] - ud["gift_time"].iloc[0]).total_seconds()/(3600*24)
        flags["risk_silent_burst"] = True if gap_day>14 else False
    else:
        flags["risk_silent_burst"] = False

    ud["hour"] = ud["gift_time"].dt.hour
    ud["is_night"] = ud["hour"].apply(lambda h:1 if h>=23 or h<=5 else 0)
    ud["is_large"] = ud["gift_value"]>large_threshold
    flags["risk_night_large"] = ((ud["is_night"]==1) & (ud["is_large"]==1)).any()
    flags["user_id"] = user_id
    return flags

def get_risk_level(row):
    """三级预警：🔴一级紧急 / 🟡二级关注 / 🟢正常"""
    f = row
    if f["risk_night_large"] and f["risk_fast_charge"]:
        return "🔴一级紧急预警"
    elif f["risk_single_large"] or f["risk_silent_burst"] or f["risk_spike"]:
        return "🟡二级关注预警"
    else:
        return "🟢正常"

# ===================== BERT弹幕风险识别【预留接口，暂不启用】=====================
# 后续队友弹幕csv准备好后，取消注释，安装transformers即可启用
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

def init_bert_model():
    """
    初始化BERT文本风险分类模型
    【预留】：后续拿到弹幕数据集后启用
    输出：tokenizer, model
    """
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    # return tokenizer, model
    return None, None

def bert_danmu_risk_predict(df_danmu, text_col="danmu_content"):
    """
    弹幕风险预测主函数
    参数：
        df_danmu: 弹幕数据集，至少包含 user_id, danmu_content（弹幕文本）
        text_col：弹幕文本列名
    返回：
        增加风险标签后的弹幕表
    """
    # =========占位逻辑，现在仅演示，不跑真实BERT=========
    df_danmu["text_risk_label"] = "待BERT模型识别"
    df_danmu["text_risk_score"] = None
    # 正式启用时，注释上面两行，解开下面BERT推理代码
    """
    tokenizer, model = init_bert_model()
    risk_result = []
    risk_score = []
    for text in df_danmu[text_col].astype(str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
            pred = out.logits.argmax(dim=1).item()
            score = torch.softmax(out.logits, dim=1)[0,1].item()
        risk_result.append("文本风险" if pred ==1 else "文本正常")
        risk_score.append(score)
    df_danmu["text_risk_label"] = risk_result
    df_danmu["text_risk_score"] = risk_score
    """
    return df_danmu

def merge_danmu_gift_risk(gift_risk_df, danmu_risk_df):
    """
    【预留】合并礼物打赏风险 + 弹幕文本风险，按user_id合并
    输出融合后的用户综合风险等级（行为+文本双维度）
    """
    merge_df = pd.merge(gift_risk_df, danmu_risk_df[["user_id","text_risk_label","text_risk_score"]],
                        on="user_id", how="left")
    # 综合风险判定规则示例，后续可以自行修改
    def get_comprehensive_risk(row):
        # 行为高风险 OR 弹幕文本风险 → 综合高风险
        if row.get("is_high_risk",0) ==1 or row.get("text_risk_label") == "文本风险":
            return "🔴综合高风险用户"
        elif row.get("risk_level") == "🟡二级关注预警":
            return "🟡综合关注用户"
        else:
            return "🟢综合正常用户"
    merge_df["comprehensive_risk"] = merge_df.apply(get_comprehensive_risk, axis=1)
    return merge_df


# # ===================== Whisper ASR语音转写【MVP预留，支持mock兜底】 =====================
# def speech_to_text(audio_file_path, use_mock:bool=True):
#     """
#     音频转文本
#     use_mock=True：不加载whisper模型，直接返回模拟转写结果，保证网页可以演示；
#     use_mock=False：真实调用whisper‑base，需要提前pip install openai‑whisper ffmpeg
#     """
#     if use_mock:
#         # Mock模拟转写结果，演示用
#         mock_text_list = [
#             "家人们礼物刷起来，给我冲冲榜！",
#             "喜欢主播的可以多点点礼物支持一下",
#             "感谢各位大哥的打赏！",
#             "没有诱导话术，正常闲聊直播内容。"
#         ]
#         import random
#         return random.choice(mock_text_list)
#     else:
#         # --------真实whisper分支，需要安装依赖后取消注释--------
#         # import whisper
#         # model = whisper.load_model("base")
#         # res = model.transcribe(audio_file_path)
#         # return res["text"]
#         pass
# ===================== Whisper ASR语音转写【真实/Mock双模式】 =====================
def speech_to_text(audio_file, use_mock: bool = True):
    """
    audio_file: streamlit上传的文件对象
    use_mock=True：模拟返回文本，不需要whisper；False使用真实whisper‑base
    """
    if use_mock:
        import random
        mock_text_list = [
            "家人们礼物刷起来，给我冲冲榜！",
            "喜欢主播的可以多点点礼物支持一下",
            "感谢各位大哥的打赏！",
            "没有诱导话术，正常闲聊直播内容。"
        ]
        return random.choice(mock_text_list)
    else:
        import tempfile
        import whisper
        # 上传的是内存字节流，存临时文件给whisper读取
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        try:
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            return result["text"]
        finally:
            os.unlink(tmp_path)





# def export_insurance_evidence(gift_risk_df, danmu_df=None, speech_text:str="", save_path:str="output/insurance_evidence.csv"):
#     """
#     【保险理赔证据链导出函数】
#     输入：行为风险表、弹幕风险表、音频转写文本
#     输出：csv证据包，留存时间线、风险标签、文本内容，用于保险举证
#     """
#     export_df = gift_risk_df.copy()
#     export_df["asr_audio_transcribe_text"] = speech_text if speech_text else "无音频记录"
#
#     if danmu_df is not None and len(danmu_df)>0:
#         merge_df = pd.merge(
#             export_df,
#             danmu_df[["user_id","text_risk_label","text_risk_score"]],
#             how="left",
#             on="user_id"
#         )
#         out = merge_df
#     else:
#         out = export_df
#
#     out.to_csv(save_path, index=False, encoding="utf‑8‑sig")
#     return save_path
import os
import zipfile
import pandas as pd

def build_user_risk_timeline(gift_df, risk_flag_df):
    """
    为每个用户构建风险事件时间线明细
    gift_df：原始送礼数据
    risk_flag_df：calc_risk_flags产出的风险标记结果
    return: 全量事件时间线DataFrame
    """
    gift_df = gift_df.copy()
    gift_df["gift_time"] = pd.to_datetime(gift_df["gift_time"])
    timeline_list = []

    for _, row in gift_df.iterrows():
        uid = row["user_id"]
        t = row["gift_time"]
        val = row["gift_value"]
        room = row["room_id"]

        item = {
            "user_id": uid,
            "event_time": t,
            "event_type": "gift_recharge",
            "room_id": room,
            "gift_value": val,
            "event_desc": f"用户打赏，金额{val}"
        }
        timeline_list.append(item)

    # 拼接风险标签描述
    timeline_df = pd.DataFrame(timeline_list)
    return timeline_df


def build_extra_evidence(gift_risk_df, raw_gift_df, sample_user_limit: int = 10):
    """
    生成4类补充证据（模拟数据，用于完善保险举证证据链）
    返回：{文件名: DataFrame}
      danmu_record.csv       弹幕记录
      chat_record.csv        私聊/客服沟通记录
      account_login_log.csv  账号登录日志
      device_info.csv        设备信息
    """
    raw = raw_gift_df.copy()
    raw["gift_time"] = pd.to_datetime(raw["gift_time"])

    # 取前N个用户作为样例，避免文件过大
    user_ids = list(dict.fromkeys(raw["user_id"].tolist()))[:sample_user_limit]

    # 1 弹幕记录（正常弹幕，作为行为留存证据）
    normal_danmu = ["哈哈哈哈", "主播好", "666", "来了来了", "这个怎么弄",
                    "主播声音好听", "学到了", "晚安"]
    danmu_rows = []
    for i, uid in enumerate(user_ids):
        udata = raw[raw["user_id"] == uid]
        room = udata["room_id"].iloc[0]
        t0 = udata["gift_time"].iloc[0]
        for j in range(2):
            danmu_rows.append({
                "user_id": uid,
                "room_id": room,
                "danmu_time": str(t0),
                "danmu_content": normal_danmu[(i + j) % len(normal_danmu)],
                "risk_flag": "正常"
            })
    danmu_record = pd.DataFrame(danmu_rows)

    # 2 私聊/客服沟通记录
    t_first = str(raw["gift_time"].iloc[0])
    t_last = str(raw["gift_time"].iloc[-1])
    chat_record = pd.DataFrame([
        {"msg_time": t_first, "sender": "用户", "receiver": "主播", "content": "主播在吗"},
        {"msg_time": t_first, "sender": "主播", "receiver": "用户", "content": "在的，欢迎"},
        {"msg_time": t_last, "sender": "监护人", "receiver": "平台客服", "content": "申请退款，孩子未成年打赏"},
        {"msg_time": t_last, "sender": "平台客服", "receiver": "监护人", "content": "请提供监护关系证明材料"}
    ])

    # 3 账号登录日志（IP脱敏）
    login_rows = []
    for uid in user_ids:
        t0 = raw[raw["user_id"] == uid]["gift_time"].iloc[0]
        login_rows.append({
            "user_id": uid,
            "login_time": str(t0),
            "login_ip_masked": "192.168.***.***",
            "device_id": "DEV-" + str(uid)[-4:],
            "login_result": "成功"
        })
    login_log = pd.DataFrame(login_rows)

    # 4 设备信息
    models = ["Xiaomi 13", "iPhone 13", "HUAWEI P60", "OPPO Reno10"]
    os_list = ["Android 13", "iOS 17", "HarmonyOS 4", "Android 14"]
    device_rows = []
    for i, uid in enumerate(user_ids):
        device_rows.append({
            "user_id": uid,
            "device_id": "DEV-" + str(uid)[-4:],
            "device_model": models[i % len(models)],
            "os": os_list[i % len(os_list)],
            "app_version": "直播平台 v8.2.1",
            "is_common_device": "是"
        })
    device_info = pd.DataFrame(device_rows)

    return {
        "danmu_record.csv": danmu_record,
        "chat_record.csv": chat_record,
        "account_login_log.csv": login_log,
        "device_info.csv": device_info
    }


def export_insurance_evidence_full(
    gift_risk_df,
    raw_gift_df,
    danmu_df=None,
    speech_text: str = "",
    output_dir: str = "output/evidence_package"
):
    """
    【强化版保险理赔证据包】
    输出：
        1.user_risk_summary.csv 用户综合风险总表
        2.user_risk_timeline.csv 用户行为事件时间线明细
        3.danmu_risk.csv 弹幕风险结果（如有弹幕）
        4.audio_transcript.txt 音频转写文本（如有音频）
        全部打包成 zip 证据包
    return：zip文件路径
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1 用户风险总表
    summary_path = os.path.join(output_dir, "user_risk_summary.csv")
    sum_df = gift_risk_df.copy()
    sum_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 2 行为时间线明细
    timeline_df = build_user_risk_timeline(raw_gift_df, gift_risk_df)
    timeline_path = os.path.join(output_dir, "user_risk_timeline.csv")
    timeline_df.to_csv(timeline_path, index=False, encoding="utf-8-sig")

    file_list = [summary_path, timeline_path]

    # 3 弹幕风险文件（如果传入弹幕数据）
    if danmu_df is not None and len(danmu_df) > 0:
        danmu_path = os.path.join(output_dir, "danmu_risk.csv")
        danmu_df.to_csv(danmu_path, index=False, encoding="utf-8-sig")
        file_list.append(danmu_path)

    # 4 音频转写文本 + 话术风险检测结果写入txt
    if speech_text and speech_text.strip() != "":
        txt_path = os.path.join(output_dir, "audio_transcript.txt")
        # 读取session里面话术风险结果
        speech_detect_res = st.session_state.get("speech_detect_result", None)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("========直播音频转写文本========\n")
            f.write(speech_text)
            f.write("\n\n========直播话术风险检测结果========\n")
            if speech_detect_res is not None:
                if speech_detect_res["whole_risk"]:
                    f.write("判定结果：本场直播【存在诱导打赏风险】\n")
                    f.write(f"风险语句数量：{len(speech_detect_res['risk_sentences'])}\n")
                    f.write("-----风险句子明细-----\n")
                    for risk_item in speech_detect_res['risk_sentences']:
                        f.write(f"句子:{risk_item['sentence']} | 置信度:{risk_item['score']}\n")
                else:
                    f.write("判定结果：本场直播【未发现诱导打赏风险】\n")
            else:
                f.write("未执行直播话术风险检测\n")
        file_list.append(txt_path)

        # 4-1 全部分句识别结果，输出结构化csv（如已执行识别）
        if speech_detect_res is not None:
            speech_risk_csv_path = os.path.join(output_dir, "speech_risk_sentences.csv")
            pd.DataFrame(speech_detect_res["all_sent_result"]).to_csv(
                speech_risk_csv_path, index=False, encoding="utf-8-sig"
            )
            file_list.append(speech_risk_csv_path)

    # 5 补充证据文件（弹幕记录/私聊记录/登录日志/设备信息，模拟数据）
    extra_evidence = build_extra_evidence(gift_risk_df, raw_gift_df)
    for ev_name, ev_df in extra_evidence.items():
        ev_path = os.path.join(output_dir, ev_name)
        ev_df.to_csv(ev_path, index=False, encoding="utf-8-sig")
        file_list.append(ev_path)

    # 6 人脸核验日志（如有核验记录）
    face_log = st.session_state.get("face_verify_log", [])
    if face_log:
        face_path = os.path.join(output_dir, "face_verify_log.txt")
        with open(face_path, "w", encoding="utf-8") as f:
            f.write("========人脸核验记录========\n")
            for item in face_log:
                f.write(
                    f"时间:{item['time']} | 用户:{item['user_id']} | 金额:{item['recharge_amount']} | "
                    f"需核验:{item['verify_required']} | 核验结果:{item['verify_result']} | "
                    f"SDK:{item.get('verify_sdk', '-')} | 置信度:{item.get('confidence', '-')} | 处置:{item['final_action']}\n"
                )
        file_list.append(face_path)

    # 打包zip
    zip_path = os.path.join(output_dir, "insurance_evidence_package.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in file_list:
            arcname = os.path.basename(fpath)
            zf.write(fpath, arcname)

    return zip_path

import re

# ----------------关键词规则版 直播话术风险识别（方案B 分句检测，MVP网页演示用）----------------
def single_sentence_rule_predict(sentence: str):
    """单句话关键词判断，替代BERT，演示用"""
    sentence = sentence.strip()
    if len(sentence) == 0:
        return {"sentence": sentence, "label": "无文本", "score": 0.0}

    risk_keywords = [
        "刷礼物","打赏","冲榜","上票","火箭","穿云箭","返你","退你","返现",
        "验证码","家长","未成年","通道","加我微信","内部","解封","防沉迷",
        "榜一","榜三","PK","帮帮","支持一下","刷单","佣金","见面","福利",
        "续费","守护","凑业绩","救我","不要告诉家长","借钱","押金"
    ]
    is_risk = any(k in sentence for k in risk_keywords)
    if is_risk:
        return {
            "sentence": sentence,
            "label": "诱导风险",
            "score": 0.9
        }
    else:
        return {
            "sentence": sentence,
            "label": "正常",
            "score": 0.1
        }


def long_text_speech_risk_detect_rule(full_text: str):
    """
    方案B：长文本按句号问号感叹号分句
    返回：
        whole_risk:bool True=本场存在诱导风险
        risk_sentences:list 风险句子
        all_sent_result:list 全部句子结果
    """
    sentences = re.split(r'[。？！]', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip())>0]
    all_sent_result = []
    risk_sentences = []
    for sent in sentences:
        res = single_sentence_rule_predict(sent)
        all_sent_result.append(res)
        if res["label"] == "诱导风险":
            risk_sentences.append(res)

    whole_risk = len(risk_sentences) > 0
    return {
        "whole_risk": whole_risk,
        "risk_sentences": risk_sentences,
        "all_sent_result": all_sent_result
    }

# ===================== BERT微调模型：直播话术真实推理（懒加载，方案B分句） =====================
_LIVE_BERT = {"tokenizer": None, "model": None}

def load_live_speech_bert(model_dir: str = "bert_live_speech_best"):
    """懒加载微调后的BERT话术模型，全局只加载一次，避免网页启动时卡顿"""
    if _LIVE_BERT["model"] is None:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import BertTokenizer, BertForSequenceClassification
        _LIVE_BERT["tokenizer"] = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        _LIVE_BERT["model"] = model
    return _LIVE_BERT["tokenizer"], _LIVE_BERT["model"]


def single_sentence_bert_predict(sentence: str, threshold: float = 0.5):
    """单句BERT推理，返回 sentence/label/score，结构与关键词版一致"""
    import torch
    sentence = sentence.strip()
    if len(sentence) == 0:
        return {"sentence": sentence, "label": "无文本", "score": 0.0}
    tokenizer, model = load_live_speech_bert()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0, 1].item()
    label = "诱导风险" if prob >= threshold else "正常"
    return {"sentence": sentence, "label": label, "score": round(prob, 4)}


def long_text_speech_risk_detect_bert(full_text: str, threshold: float = 0.5):
    """
    方案B：长文本按句号/问号/感叹号分句，每句单独BERT推理，
    任一句判定为诱导风险，则整场直播标记存在诱导风险。
    返回结构与 long_text_speech_risk_detect_rule 完全一致，便于网页切换。
    """
    sentences = re.split(r'[。？！]', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    all_sent_result = []
    risk_sentences = []
    for sent in sentences:
        res = single_sentence_bert_predict(sent, threshold)
        all_sent_result.append(res)
        if res["label"] == "诱导风险":
            risk_sentences.append(res)
    whole_risk = len(risk_sentences) > 0
    return {
        "whole_risk": whole_risk,
        "risk_sentences": risk_sentences,
        "all_sent_result": all_sent_result
    }

# ===================== 多方责任划分判定引擎（保险理赔核心） =====================

def calculate_responsibility(
    loss_amount: float,
    has_anchor_induce: bool = False,
    induce_level: str = "none",       # none / mild / severe
    has_guardian_neglect: bool = False,
    neglect_level: str = "none",      # none / mild / severe
    platform_refund: float = 0.0,
    is_minor_self: bool = True
):
    """
    多方责任划分判定函数
    输入：
        loss_amount: 打赏损失总金额
        has_anchor_induce: 主播是否存在诱导行为
        induce_level: 诱导程度 none/mild/severe
        has_guardian_neglect: 监护人是否存在失职
        neglect_level: 失职程度 none/mild/severe
        platform_refund: 平台已退款金额
        is_minor_self: 是否为未成年人自发行为
    输出：
        dict: 责任方、各责任比例、保险建议赔付比例、判定说明、可赔金额
    """
    result = {
        "responsibility_parties": [],   # 责任方列表
        "anchor_ratio": 0.0,            # 主播责任比例
        "guardian_ratio": 0.0,          # 监护人责任比例
        "platform_ratio": 0.0,          # 平台责任比例
        "minor_ratio": 0.0,             # 未成年人自发比例
        "insurance_pay_ratio": 0.0,     # 保险建议赔付比例
        "insurance_pay_amount": 0.0,    # 保险建议赔付金额（扣平台退款后）
        "verdict": "",                  # 判定结论
        "detail": []                    # 判定明细说明
    }

    # ---------- 规则1：主播刻意诱导 → 主播全责，保险拒赔 ----------
    if has_anchor_induce and induce_level == "severe":
        result["anchor_ratio"] = 1.0
        result["responsibility_parties"].append("主播（刻意诱导）")
        result["insurance_pay_ratio"] = 0.0
        result["verdict"] = "主播刻意诱导消费，主播承担全部责任，保险不予赔付"
        result["detail"].append("经AI风控核验，直播间存在刻意诱导未成年人打赏行为")
        result["detail"].append("依据保险条款责任免除第4条，保险人不承担赔偿责任")
        result["insurance_pay_amount"] = 0.0
        return result

    # ---------- 规则2：主播轻微诱导 → 主播部分责任，保险扣减 ----------
    if has_anchor_induce and induce_level == "mild":
        result["anchor_ratio"] = 0.4
        result["responsibility_parties"].append("主播（轻微诱导话术）")
        result["detail"].append("直播间存在轻微诱导话术，主播承担40%责任")

    # ---------- 规则3：监护人失职 ----------
    if has_guardian_neglect:
        if neglect_level == "severe":
            result["guardian_ratio"] = 0.5
            result["responsibility_parties"].append("监护人（严重失职）")
            result["detail"].append("监护人未开启未成年人模式、长期放任设备使用，承担50%责任")
        elif neglect_level == "mild":
            result["guardian_ratio"] = 0.3
            result["responsibility_parties"].append("监护人（一般失职）")
            result["detail"].append("监护人存在监护疏忽，承担30%责任")

    # ---------- 规则4：平台已退款 ----------
    if platform_refund > 0:
        refund_ratio = min(platform_refund / loss_amount, 1.0) if loss_amount > 0 else 0
        result["platform_ratio"] = round(refund_ratio, 4)
        result["responsibility_parties"].append("直播平台（已退款）")
        result["detail"].append(f"平台已退款{platform_refund:.2f}元，占损失{refund_ratio:.1%}")

    # ---------- 规则5：剩余比例归为未成年人自发 ----------
    assigned = result["anchor_ratio"] + result["guardian_ratio"] + result["platform_ratio"]
    remaining = max(0.0, 1.0 - assigned)
    if remaining > 0 and is_minor_self:
        result["minor_ratio"] = round(remaining, 4)
        result["responsibility_parties"].append("未成年人（自发非理性消费）")
        result["detail"].append(f"剩余{remaining:.1%}为未成年人自发非理性消费")

    # ---------- 保险赔付比例 = 未成年人自发比例（保险只赔自发部分） ----------
    result["insurance_pay_ratio"] = result["minor_ratio"]

    # ---------- 可赔金额 = (损失 - 平台已退款) × 保险赔付比例 ----------
    remaining_loss = max(0.0, loss_amount - platform_refund)
    result["insurance_pay_amount"] = round(remaining_loss * result["insurance_pay_ratio"], 2)

    # ---------- 判定结论 ----------
    if result["insurance_pay_ratio"] >= 0.9:
        result["verdict"] = "未成年人自发非理性消费，无主播诱导、监护人尽责，建议正常赔付"
    elif result["insurance_pay_ratio"] > 0:
        result["verdict"] = f"存在多方责任，保险按{result['insurance_pay_ratio']:.0%}比例赔付"
    else:
        result["verdict"] = "存在责任免除情形，保险不予赔付"

    return result

# ===================== 赔款计算器（C端家庭保险） =====================

def calculate_compensation(
    loss_amount: float,
    family_risk_level: str = "medium",   # low / medium / high
    platform_refund: float = 0.0,
    responsibility_ratio: float = 1.0,
    single_limit: float = 5000.0,
    annual_limit: float = 5000.0,
    annual_paid: float = 0.0
):
    """
    C端家庭保险赔款计算器
    规则（对照产品文档）：
      1. 免赔额：低风险200/中风险300/高风险500，或损失金额10%，二者取高
      2. 先扣平台已退款
      3. 再扣免赔额
      4. 再乘责任比例
      5. 不超单次事故限额5000，不超年度累计限额5000
    输入：
      loss_amount: 打赏损失总金额
      family_risk_level: 家庭风险等级 low/medium/high
      platform_refund: 平台已退款金额
      responsibility_ratio: 保险赔付比例（来自calculate_responsibility的insurance_pay_ratio）
      single_limit: 单次事故赔偿限额，默认5000
      annual_limit: 年度累计赔偿限额，默认5000
      annual_paid: 本年度已赔付金额
    输出：
      dict: 各步骤金额、最终赔付、计算明细
    """
    result = {
        "loss_amount": round(loss_amount, 2),
        "platform_refund": round(platform_refund, 2),
        "deductible": 0.0,
        "after_refund": 0.0,
        "after_deductible": 0.0,
        "ratio_payout": 0.0,
        "after_single_limit": 0.0,
        "annual_remaining": 0.0,
        "final_payout": 0.0,
        "detail": []
    }

    # ---------- 步骤1：计算免赔额 ----------
    deductible_base = {"low": 200, "medium": 300, "high": 500}.get(family_risk_level, 300)
    deductible_by_pct = loss_amount * 0.10
    deductible = max(deductible_base, deductible_by_pct)
    result["deductible"] = round(deductible, 2)
    result["detail"].append(
        "免赔额：基础" + str(deductible_base) + "元 与 损失10%(" + str(round(deductible_by_pct,2)) + "元) 取高 = " + str(round(deductible,2)) + "元"
    )

    # ---------- 步骤2：扣平台已退款 ----------
    after_refund = max(0.0, loss_amount - platform_refund)
    result["after_refund"] = round(after_refund, 2)
    result["detail"].append(
        "扣除平台退款" + str(round(platform_refund,2)) + "元后：" + str(round(after_refund,2)) + "元"
    )

    # ---------- 步骤3：扣免赔额 ----------
    after_deductible = max(0.0, after_refund - deductible)
    result["after_deductible"] = round(after_deductible, 2)
    result["detail"].append(
        "扣除免赔额" + str(round(deductible,2)) + "元后：" + str(round(after_deductible,2)) + "元"
    )

    # ---------- 步骤4：乘责任比例 ----------
    ratio_payout = after_deductible * responsibility_ratio
    result["ratio_payout"] = round(ratio_payout, 2)
    result["detail"].append(
        "乘以责任比例" + str(round(responsibility_ratio*100,1)) + "%：" + str(round(ratio_payout,2)) + "元"
    )

    # ---------- 步骤5：单次限额 ----------
    after_single = min(ratio_payout, single_limit)
    result["after_single_limit"] = round(after_single, 2)
    if ratio_payout > single_limit:
        result["detail"].append(
            "超单次限额" + str(single_limit) + "元，扣减至" + str(single_limit) + "元"
        )

    # ---------- 步骤6：年度限额 ----------
    annual_remaining = max(0.0, annual_limit - annual_paid)
    result["annual_remaining"] = round(annual_remaining, 2)
    final_payout = min(after_single, annual_remaining)
    result["final_payout"] = round(final_payout, 2)
    if after_single > annual_remaining:
        result["detail"].append(
            "年度剩余额度" + str(round(annual_remaining,2)) + "元，最终赔付" + str(round(final_payout,2)) + "元"
        )
    else:
        result["detail"].append("最终赔付金额：" + str(round(final_payout,2)) + "元")

    return result


# ===================== 一键理赔判定（任务2+任务3合并） =====================

def process_claim(
    loss_amount: float,
    family_risk_level: str = "medium",
    has_anchor_induce: bool = False,
    induce_level: str = "none",
    has_guardian_neglect: bool = False,
    neglect_level: str = "none",
    platform_refund: float = 0.0,
    annual_paid: float = 0.0
):
    """
    一键理赔判定：自动串联责任划分 + 赔款计算
    输入案件信息，输出完整理赔结论
    """
    # 第一步：调用任务2 — 责任划分
    resp = calculate_responsibility(
        loss_amount=loss_amount,
        has_anchor_induce=has_anchor_induce,
        induce_level=induce_level,
        has_guardian_neglect=has_guardian_neglect,
        neglect_level=neglect_level,
        platform_refund=platform_refund,
        is_minor_self=True
    )

    # 第二步：调用任务3 — 赔款计算（把任务2的赔付比例传进去）
    comp = calculate_compensation(
        loss_amount=loss_amount,
        family_risk_level=family_risk_level,
        platform_refund=platform_refund,
        responsibility_ratio=resp["insurance_pay_ratio"],
        annual_paid=annual_paid
    )

    # 第三步：合并输出
    return {
        "loss_amount": loss_amount,
        "responsibility": resp,
        "compensation": comp,
        "final_verdict": resp["verdict"],
        "final_payout": comp["final_payout"],
        "is_approved": comp["final_payout"] > 0
    }

# ===================== 承保定价引擎 =====================

def assess_family_risk(
    minor_mode_enabled: bool = True,
    historical_claims: int = 0,
    child_age: int = 12
):
    """
    家庭风险评级
    输入：是否开启未成年模式、历史理赔次数、孩子年龄
    输出：risk_level (low/medium/high) + 评级说明
    """
    score = 0
    reasons = []

    # 因素1：未成年模式
    if not minor_mode_enabled:
        score += 2
        reasons.append("未开启平台未成年人守护模式")
    else:
        reasons.append("已开启平台未成年人守护模式")

    # 因素2：历史理赔
    if historical_claims >= 2:
        score += 2
        reasons.append("历史理赔" + str(historical_claims) + "次，风险较高")
    elif historical_claims == 1:
        score += 1
        reasons.append("历史理赔1次")
    else:
        reasons.append("无历史理赔记录")

    # 因素3：孩子年龄段（10-16岁为高风险段）
    if 10 <= child_age <= 16:
        score += 1
        reasons.append("孩子处于10-16岁高风险年龄段")
    else:
        reasons.append("孩子不在高风险年龄段")

    # 评级
    if score >= 3:
        level = "high"
        level_text = "高风险"
    elif score >= 1:
        level = "medium"
        level_text = "中风险"
    else:
        level = "low"
        level_text = "低风险"

    return {
        "risk_level": level,
        "risk_level_text": level_text,
        "risk_score": score,
        "reasons": reasons
    }


def assess_anchor_risk(
    induce_risk_score: float = 0.0,
    complaint_count: int = 0,
    minor_fan_ratio: float = 0.1
):
    """
    主播风险评级
    输入：诱导话术风险分(0-1)、历史被投诉次数、粉丝中未成年人占比
    输出：risk_level + 评级说明
    """
    score = 0
    reasons = []

    # 因素1：诱导话术风险分
    if induce_risk_score >= 0.7:
        score += 2
        reasons.append("直播间诱导话术风险分" + str(round(induce_risk_score, 2)) + "，高风险")
    elif induce_risk_score >= 0.3:
        score += 1
        reasons.append("直播间诱导话术风险分" + str(round(induce_risk_score, 2)) + "，中等风险")
    else:
        reasons.append("直播间诱导话术风险分" + str(round(induce_risk_score, 2)) + "，低风险")

    # 因素2：投诉次数
    if complaint_count >= 3:
        score += 2
        reasons.append("历史被投诉" + str(complaint_count) + "次")
    elif complaint_count >= 1:
        score += 1
        reasons.append("历史被投诉" + str(complaint_count) + "次")
    else:
        reasons.append("无投诉记录")

    # 因素3：未成年人粉丝占比
    if minor_fan_ratio >= 0.3:
        score += 1
        reasons.append("粉丝中未成年人占比" + str(round(minor_fan_ratio*100, 1)) + "%，较高")
    else:
        reasons.append("粉丝中未成年人占比" + str(round(minor_fan_ratio*100, 1)) + "%")

    if score >= 3:
        level = "high"
        level_text = "高风险"
    elif score >= 1:
        level = "medium"
        level_text = "中风险"
    else:
        level = "low"
        level_text = "低风险"

    return {
        "risk_level": level,
        "risk_level_text": level_text,
        "risk_score": score,
        "reasons": reasons
    }


def calculate_premium(customer_type: str = "family", risk_level: str = "medium"):
    """
    承保定价：根据客户类型和风险等级输出保费方案
    customer_type: family / anchor
    risk_level: low / medium / high
    """
    pricing_table = {
        "family": {
            "low":    {"premium": 95,  "deductible": 200,  "coverage": 5000},
            "medium": {"premium": 120, "deductible": 300,  "coverage": 5000},
            "high":   {"premium": 160, "deductible": 500,  "coverage": 5000},
        },
        "anchor": {
            "low":    {"premium": 140, "deductible": 800,  "coverage": 20000},
            "medium": {"premium": 180, "deductible": 1000, "coverage": 20000},
            "high":   {"premium": 230, "deductible": 1500, "coverage": 20000},
        }
    }

    type_text = "C端家庭" if customer_type == "family" else "B端主播"
    level_text = {"low": "低风险", "medium": "中风险", "high": "高风险"}[risk_level]
    plan = pricing_table[customer_type][risk_level]

    return {
        "customer_type": customer_type,
        "customer_type_text": type_text,
        "risk_level": risk_level,
        "risk_level_text": level_text,
        "annual_premium": plan["premium"],
        "deductible": plan["deductible"],
        "coverage": plan["coverage"],
        "plan_name": type_text + level_text + "保障计划"
    }

# ===================== 消费理性评估模型（任务1新增） =====================

def assess_consumption_rationality(raw_df, user_feat):
    """
    消费理性评估：量化未成年人消费理性程度
    输入：
      raw_df: 原始送礼数据（含 user_id / gift_time / gift_value）
      user_feat: 用户级特征表（detect_user_risk 或 build_user_feature 的输出）
    输出：
      在 user_feat 基础上新增三列：
        rationality_score  理性评分 0-100
        rationality_level  等级（理性/一般/非理性）
        rationality_reasons 扣分原因列表
    规则：
      基础分 100，按上课时段占比 / 小额多笔占比 / 周末假期占比 / 夜间占比逐项扣分
    """
    import pandas as pd

    df = raw_df.copy()
    df["gift_time"] = pd.to_datetime(df["gift_time"])
    df["hour"] = df["gift_time"].dt.hour
    df["weekday"] = df["gift_time"].dt.weekday  # 0=周一 ... 6=周日
    df["is_school_time"] = ((df["weekday"] < 5) & (df["hour"] >= 8) & (df["hour"] < 17)).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_small"] = (df["gift_value"] < 100).astype(int)
    df["school_time_money"] = df.apply(lambda r: r["gift_value"] if r["is_school_time"] == 1 else 0, axis=1)
    df["weekend_money"] = df.apply(lambda r: r["gift_value"] if r["is_weekend"] == 1 else 0, axis=1)

    # ---------- 心智特征聚合（按用户） ----------
    feat = df.groupby("user_id").agg(
        total_money=("gift_value", "sum"),
        total_cnt=("gift_value", "count"),
        small_cnt=("is_small", "sum"),
        school_time_money=("school_time_money", "sum"),
        weekend_money=("weekend_money", "sum")
    ).reset_index()

    feat["school_time_ratio"] = feat.apply(
        lambda r: r["school_time_money"] / r["total_money"] if r["total_money"] > 0 else 0, axis=1)
    feat["small_multi_ratio"] = feat.apply(
        lambda r: r["small_cnt"] / r["total_cnt"] if r["total_cnt"] > 0 else 0, axis=1)
    feat["weekend_holiday_ratio"] = feat.apply(
        lambda r: r["weekend_money"] / r["total_money"] if r["total_money"] > 0 else 0, axis=1)

    # ---------- 合并到用户特征表 ----------
    out = user_feat.copy()
    out = pd.merge(
        out,
        feat[["user_id", "school_time_ratio", "small_multi_ratio", "weekend_holiday_ratio"]],
        on="user_id", how="left")
    out["school_time_ratio"] = out["school_time_ratio"].fillna(0)
    out["small_multi_ratio"] = out["small_multi_ratio"].fillna(0)
    out["weekend_holiday_ratio"] = out["weekend_holiday_ratio"].fillna(0)

    # ---------- 规则打分 ----------
    score_list, level_list, reason_list = [], [], []
    for _, row in out.iterrows():
        score = 100
        reasons = []

        school_r = row["school_time_ratio"]
        if school_r > 0.3:
            score -= 35
            reasons.append("上课时段打赏占比" + str(round(school_r * 100, 1)) + "%，高度疑似在校时间使用")
        elif school_r > 0.1:
            score -= 20
            reasons.append("上课时段打赏占比" + str(round(school_r * 100, 1)) + "%，存在非理性消费特征")

        small_r = row["small_multi_ratio"]
        if small_r > 0.6:
            score -= 25
            reasons.append("小额多笔占比" + str(round(small_r * 100, 1)) + "%，呈隐蔽式累积消费特征")
        elif small_r > 0.3:
            score -= 15
            reasons.append("小额多笔占比" + str(round(small_r * 100, 1)) + "%，存在小额多笔累积")

        week_r = row["weekend_holiday_ratio"]
        if week_r > 0.8:
            score -= 20
            reasons.append("周末假期打赏占比" + str(round(week_r * 100, 1)) + "%，集中在休息时段")
        elif week_r > 0.5:
            score -= 10
            reasons.append("周末假期打赏占比" + str(round(week_r * 100, 1)) + "%，休息时段相对集中")

        night_r = row.get("night_ratio", 0) or 0
        if night_r > 0.5:
            score -= 20
            reasons.append("夜间打赏占比" + str(round(night_r * 100, 1)) + "%，深夜活跃")
        elif night_r > 0.2:
            score -= 10
            reasons.append("夜间打赏占比" + str(round(night_r * 100, 1)) + "%")

        score = max(0, min(100, score))
        if score >= 80:
            level = "理性"
        elif score >= 60:
            level = "一般"
        else:
            level = "非理性"

        score_list.append(score)
        level_list.append(level)
        reason_list.append(reasons)

    out["rationality_score"] = score_list
    out["rationality_level"] = level_list
    out["rationality_reasons"] = reason_list
    return out

# ===================== 身份冒用/虚假索赔检测（任务2新增） =====================

def detect_identity_spoofing(case_feat: dict):
    """
    身份冒用/虚假索赔检测
    输入（理赔调查阶段的证据特征）：
      night_ratio:      凌晨(23:00-05:00)打赏金额占比 0-1
      school_ratio:     上课时段(工作日8:00-17:00)打赏金额占比 0-1
      delete_record:    删除支付/消费记录次数
      habit_shift:      账号作息突变（1=是，0=否）
      face_verify_fail: 人脸核验失败次数
    输出：
      spoofing_score:   冒用风险评分 0-100
      spoofing_level:   正常 / 可疑 / 高风险冒用
      is_manual_review: 是否需人工重点核验（高风险或人脸多次失败时为True）
      reasons:          风险原因列表
    规则：
      基础分100，按上述特征逐项扣分
    """
    night_ratio = case_feat.get("night_ratio", 0) or 0
    school_ratio = case_feat.get("school_ratio", 0) or 0
    delete_record = case_feat.get("delete_record", 0) or 0
    habit_shift = case_feat.get("habit_shift", 0) or 0
    face_verify_fail = case_feat.get("face_verify_fail", 0) or 0

    score = 100
    reasons = []

    # ---------- 凌晨活跃 ----------
    if night_ratio > 0.6:
        score -= 30
        reasons.append("凌晨时段打赏占比" + str(round(night_ratio * 100, 1)) + "%，高度疑似未成年人深夜使用")
    elif night_ratio > 0.3:
        score -= 15
        reasons.append("凌晨时段打赏占比" + str(round(night_ratio * 100, 1)) + "%，存在深夜活跃")

    # ---------- 上课时段活跃 ----------
    if school_ratio > 0.3:
        score -= 25
        reasons.append("上课时段打赏占比" + str(round(school_ratio * 100, 1)) + "%，疑似在校时间使用")
    elif school_ratio > 0.1:
        score -= 10
        reasons.append("上课时段打赏占比" + str(round(school_ratio * 100, 1)) + "%，存在异常时段消费")

    # ---------- 删除记录 ----------
    if delete_record >= 2:
        score -= 25
        reasons.append("删除支付/消费记录" + str(delete_record) + "次，存在隐瞒消费痕迹")
    elif delete_record >= 1:
        score -= 15
        reasons.append("存在删除支付记录行为")

    # ---------- 账号作息突变 ----------
    if habit_shift == 1:
        score -= 20
        reasons.append("账号消费作息突然改变，疑似实际操作人变更")

    # ---------- 人脸核验失败 ----------
    if face_verify_fail >= 2:
        score -= 30
        reasons.append("人脸核验失败" + str(face_verify_fail) + "次，身份一致性存疑")
    elif face_verify_fail >= 1:
        score -= 15
        reasons.append("人脸核验失败1次")

    score = max(0, min(100, score))
    if score >= 80:
        level = "正常"
    elif score >= 60:
        level = "可疑"
    else:
        level = "高风险冒用"

    is_manual_review = (level == "高风险冒用") or (face_verify_fail >= 2)

    return {
        "spoofing_score": score,
        "spoofing_level": level,
        "is_manual_review": is_manual_review,
        "reasons": reasons
    }

# ===================== 复投监控（任务4新增） =====================

def monitor_reinvestment(gift_df, closed_claims):
    """
    复投监控：理赔结案后账号再次打赏预警
    输入：
      gift_df: 原始送礼数据（含 user_id / gift_time / gift_value）
      closed_claims: 已结案理赔记录列表，每项含
        case_id / account_user_id / close_time("YYYY-MM-DD") / close_result("赔付"/"拒赔")
    输出：
      DataFrame：每案一行，含复投笔数/复投金额/首笔复投时间/是否预警
    预警规则：结案后再次打赏 且 累计复投金额>=300元，标记预警
    """
    import pandas as pd

    df = gift_df.copy()
    df["gift_time"] = pd.to_datetime(df["gift_time"])

    rows = []
    for claim in closed_claims:
        uid = claim.get("account_user_id")
        close_time = pd.to_datetime(claim.get("close_time"))
        close_result = claim.get("close_result", "赔付")

        after = df[(df["user_id"] == uid) & (df["gift_time"] > close_time)]
        reinvest_cnt = len(after)
        reinvest_money = float(after["gift_value"].sum()) if reinvest_cnt > 0 else 0.0
        first_after = after["gift_time"].min() if reinvest_cnt > 0 else None
        warning = True if (reinvest_cnt > 0 and reinvest_money >= 300) else False

        rows.append({
            "case_id": claim.get("case_id", "-"),
            "account_user_id": uid,
            "close_time": close_time.strftime("%Y-%m-%d"),
            "close_result": close_result,
            "reinvest_cnt": reinvest_cnt,
            "reinvest_money": round(reinvest_money, 2),
            "first_reinvest_time": first_after.strftime("%Y-%m-%d %H:%M") if first_after is not None else "无",
            "warning": warning
        })

    return pd.DataFrame(rows)

# ===================== 理赔审核分级（任务6新增） =====================

def classify_review_grade(case_dict, spoof_result=None):
    """
    理赔审核分级
    输入：
      case_dict: 案件要素（loss_amount / family_risk_level / induce_level）
      spoof_result: 身份冒用检测输出（detect_identity_spoofing 的结果，可选）
    输出：
      grade: A / B / C
      grade_name: 自动通过 / 常规审核 / 人工重点审核
      audit_hours: 建议处理时限（小时）
      rules: 触发规则说明列表
    规则：
      C级：身份冒用需人工核验，或 损失>=3000，或 家庭高风险，或 刻意诱导
      B级：损失>=1000，或 家庭中风险，或 轻微诱导
      A级：其余低风险小额案件
    """
    loss = case_dict.get("loss_amount", 0) or 0
    fam_risk = case_dict.get("family_risk_level", "low")
    induce = case_dict.get("induce_level", "none")
    rules = []

    manual = False
    if spoof_result is not None and spoof_result.get("is_manual_review"):
        manual = True
        rules.append("身份冒用核验标记需人工复核")

    if manual or loss >= 3000 or fam_risk == "high" or induce == "severe":
        grade = "C"
        grade_name = "人工重点审核"
        audit_hours = 24
        if loss >= 3000:
            rules.append("损失金额" + str(loss) + "元，达人工重点线")
        if fam_risk == "high":
            rules.append("家庭风险等级为高风险")
        if induce == "severe":
            rules.append("主播刻意诱导")
    elif loss >= 1000 or fam_risk == "medium" or induce == "mild":
        grade = "B"
        grade_name = "常规审核"
        audit_hours = 8
        if loss >= 1000:
            rules.append("损失金额" + str(loss) + "元，常规审核线")
        if fam_risk == "medium":
            rules.append("家庭风险等级为中风险")
        if induce == "mild":
            rules.append("主播轻微诱导")
    else:
        grade = "A"
        grade_name = "自动通过"
        audit_hours = 2
        rules.append("低风险小额案件，走快速通道")

    return {
        "grade": grade,
        "grade_name": grade_name,
        "audit_hours": audit_hours,
        "rules": rules
    }

# ===================== 人脸核验SDK封装（任务7新增） =====================

def face_verify_sdk(user_id, simulate=True, pass_rate=0.8):
    """
    人脸核验SDK封装（模拟/真实对接）
    输入：
      user_id: 待核验账号
      simulate: True=模拟核验（演示用）；False=真实SDK对接点（需接入后启用）
      pass_rate: 模拟通过概率 0-1
    输出：
      passed / confidence / sdk_name / message
    真实对接说明：
      生产环境可对接平台实名人脸SDK（如腾讯慧眼/阿里实人认证）或公安/运营商实名核验接口，
      返回人脸比对分数与活体检测结果；本函数保留 simulate 开关便于演示与回归。
    """
    import random

    if simulate:
        passed = random.random() < pass_rate
        if passed:
            confidence = round(random.uniform(0.90, 0.99), 2)
            message = "活体检测通过，人脸比对一致"
        else:
            confidence = round(random.uniform(0.30, 0.55), 2)
            message = "人脸比对不一致或活体检测未通过"
        return {
            "passed": passed,
            "confidence": confidence,
            "sdk_name": "face_verify_sdk(mock)",
            "message": message
        }
    else:
        # -------- 真实SDK对接点（示例，需按所选服务商接入） --------
        # 例：腾讯云人脸核验 createFaceVerifyToken + GetFaceVerifyResult
        #     或 阿里云实人认证 DescribeFaceVerify
        # 此处仅占位，接入后返回真实结果
        return {
            "passed": None,
            "confidence": None,
            "sdk_name": "real_sdk_pending",
            "message": "真实SDK未接入，请配置服务商密钥后启用"
        }
