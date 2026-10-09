import streamlit as st
import pandas as pd
from user_risk_func import monitor_reinvestment

st.set_page_config(page_title="风险档案", layout="wide")

st.title("风险档案与投保限制管理")
st.caption("汇总家庭（用户）与主播的历史风险记录、违规次数，并给出承保限制状态")

result_df = st.session_state.get("result_df", None)
raw_df = st.session_state.get("df_data", None)

flag_cols = ["risk_fast_charge", "risk_single_large", "risk_spike",
             "risk_silent_burst", "risk_night_large"]

def user_level(row):
    if row["is_high_risk"] == 1:
        return "高风险"
    if any(str(row[c]) == "存在风险" for c in flag_cols):
        return "中风险"
    return "正常"

def level_with_icon(level):
    return {"高风险": "🔴高风险", "中风险": "🟡中风险", "正常": "🟢正常"}[level]

def family_underwriting(level, violation_cnt):
    if level == "高风险" or violation_cnt >= 3:
        return "拒保"
    elif level == "中风险" or violation_cnt >= 1:
        return "加费承保"
    else:
        return "标准承保"

def anchor_underwriting(level):
    return {"high": "拒保", "medium": "加费承保", "low": "标准承保"}[level]

if result_df is None:
    st.warning("尚未检测到风控数据。")
    st.markdown("请先到【平台风控后台】上传直播打赏csv文件并运行检测，再查看风险档案。")
    st.stop()

rdf = result_df.copy()
rdf["风险等级"] = rdf.apply(user_level, axis=1)
# 违规次数 = 命中风险类型数量
rdf["违规次数"] = rdf[flag_cols].apply(lambda r: sum(str(v) == "存在风险" for v in r), axis=1)

tab_family, tab_anchor, tab_reinvest = st.tabs(["家庭（用户）风险档案", "主播风险档案", "复投监控"])

# ===================== Tab1 家庭风险档案 =====================
with tab_family:
    n_total = len(rdf)
    n_high = (rdf["风险等级"] == "高风险").sum()
    family_rows = []
    for _, r in rdf.iterrows():
        status = family_underwriting(r["风险等级"], r["违规次数"])
        family_rows.append({
            "用户（ID脱敏）": "****" + str(r["user_id"])[-4:],
            "风险等级": level_with_icon(r["风险等级"]),
            "累计打赏金额": r["total_gift_money"],
            "打赏笔数": r["total_gift_cnt"],
            "违规次数": r["违规次数"],
            "投保状态": status
        })
    family_table = pd.DataFrame(family_rows)
    n_reject = (family_table["投保状态"] == "拒保").sum()
    n_extra = (family_table["投保状态"] == "加费承保").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("档案用户总数", n_total)
    c2.metric("高风险用户", n_high)
    c3.metric("加费承保", n_extra)
    c4.metric("拒保", n_reject)

    st.divider()
    st.dataframe(family_table, use_container_width=True)
    st.caption("承保规则：高风险或违规≥3次 拒保；中风险或违规≥1次 加费承保；其余标准承保")

    st.divider()
    st.subheader("查看单用户风险档案")
    user_list_arch = sorted(rdf["user_id"].tolist())
    default_arch = 0
    jump_from_platform = st.session_state.get("guard_jump_user", None)
    if jump_from_platform in user_list_arch:
        default_arch = user_list_arch.index(jump_from_platform)
    jump_user = st.selectbox("选择用户", user_list_arch, key="arch_jump_user", index=default_arch)
    if st.button("跳转查看该用户档案详情", key="btn_jump_guardian"):
        st.session_state["guard_jump_user"] = jump_user
        st.switch_page("pages/2_监护人守护页面.py")
    st.caption("跳转后在监护人页面查看该账号的消费总览、消费理性评估与风险行为标签")

# ===================== Tab2 主播风险档案 =====================
with tab_anchor:
    if raw_df is None:
        st.warning("缺少原始送礼数据，无法按直播间统计。")
    else:
        merged = raw_df.merge(rdf[["user_id", "风险等级"]], on="user_id", how="left")
        # 用户-直播间去重后统计观众与高风险观众
        ur = merged[["room_id", "user_id", "风险等级"]].drop_duplicates()
        ur["is_high"] = (ur["风险等级"] == "高风险").astype(int)
        room_aud = ur.groupby("room_id").agg(
            观众数=("user_id", "nunique"),
            高风险观众=("is_high", "sum")
        )
        room_money = merged.groupby("room_id").agg(累计打赏=("gift_value", "sum"))
        room_stat = room_aud.join(room_money)

        from user_risk_func import assess_anchor_risk
        anchor_rows = []
        for room_id, r in room_stat.iterrows():
            risk_aud = int(r["高风险观众"])
            aud = int(r["观众数"])
            ratio = risk_aud / aud if aud > 0 else 0
            assess = assess_anchor_risk(
                induce_risk_score=ratio,       # 以高风险观众占比作为话术风险代理
                complaint_count=risk_aud,      # 模拟投诉线索数
                minor_fan_ratio=ratio          # 以风险观众占比估算未成年粉丝占比
            )
            anchor_rows.append({
                "直播间ID": room_id,
                "观众数": aud,
                "高风险观众": risk_aud,
                "累计打赏": r["累计打赏"],
                "话术风险分": round(ratio, 2),
                "主播评级": assess["risk_level_text"],
                "投保状态": anchor_underwriting(assess["risk_level"])
            })
        anchor_table = pd.DataFrame(anchor_rows)
        a1, a2, a3 = st.columns(3)
        a1.metric("直播间总数", len(anchor_table))
        a2.metric("高评级主播", (anchor_table["主播评级"] == "高风险").sum())
        a3.metric("拒保直播间", (anchor_table["投保状态"] == "拒保").sum())
        st.divider()
        st.dataframe(anchor_table, use_container_width=True)
        st.caption("说明：话术风险分与未成年粉丝占比以该直播间高风险观众占比估算，投诉线索数为模拟值；主播评级由 assess_anchor_risk 计算")

# ===================== 复投监控（任务4新增） =====================
with tab_reinvest:
    st.markdown("监控理赔结案账号在结案后是否再次打赏，防止理赔后继续消费与反复套利")

    if raw_df is None:
        st.warning("缺少原始送礼数据，无法进行复投监控")
    else:
        gift_for_monitor = raw_df.copy()
        gift_for_monitor["gift_time"] = pd.to_datetime(gift_for_monitor["gift_time"])
        t_min = gift_for_monitor["gift_time"].min()
        t_max = gift_for_monitor["gift_time"].max()

        # 结案记录：优先使用理赔工作台写入的结案记录，否则用模拟记录演示
        closed = st.session_state.get("closed_claims", [])
        if closed:
            st.caption("当前使用理赔工作台的结案记录，共" + str(len(closed)) + "条")
            claims_input = closed
        else:
            st.caption("尚未在理赔工作台结案案件，以下为模拟结案记录（关联数据集真实账号）演示")
            user_list = sorted(gift_for_monitor["user_id"].unique().tolist())
            claims_input = []
            if len(user_list) >= 1:
                claims_input.append({"case_id": "MOCK-001", "account_user_id": user_list[0],
                                     "close_time": t_min.strftime("%Y-%m-%d"), "close_result": "赔付"})
            if len(user_list) >= 2:
                claims_input.append({"case_id": "MOCK-002", "account_user_id": user_list[1],
                                     "close_time": t_max.strftime("%Y-%m-%d"), "close_result": "拒赔"})
            if len(user_list) >= 3:
                claims_input.append({"case_id": "MOCK-003", "account_user_id": user_list[2],
                                     "close_time": (t_min + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "close_result": "赔付"})

        if claims_input:
            monitor_df = monitor_reinvestment(gift_for_monitor, claims_input)
            warn_n = int(monitor_df["warning"].sum())
            c1, c2 = st.columns(2)
            c1.metric("已结案案件", len(monitor_df))
            c2.metric("复投预警", warn_n)
            st.dataframe(monitor_df, width="stretch", hide_index=True)
            if warn_n > 0:
                st.error("存在" + str(warn_n) + "个账号理赔结案后再次打赏，建议立即冻结额度或人工回访")
            else:
                st.success("未发现复投行为")
        else:
            st.info("暂无结案记录")
