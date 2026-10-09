import streamlit as st
from user_risk_func import assess_family_risk, assess_anchor_risk, calculate_premium

st.set_page_config(page_title="承保定价台", layout="wide")
st.title("承保定价台")
st.markdown("风险评级 → 差异化保费 / 免赔额 / 保额 自动定价")

# 选择客户类型
customer_type = st.radio(
    "选择客户类型",
    options=["C端家庭", "B端主播"],
    horizontal=True
)
ctype = "family" if customer_type == "C端家庭" else "anchor"

st.divider()

# ===================== C端家庭风险评级 =====================
if ctype == "family":
    st.subheader("家庭风险因素录入")
    col1, col2, col3 = st.columns(3)
    with col1:
        minor_mode = st.checkbox("已开启平台未成年人守护模式", value=True)
    with col2:
        hist_claims = st.number_input("历史理赔次数", min_value=0, max_value=10, value=0, step=1)
    with col3:
        child_age = st.number_input("被保险人年龄", min_value=0, max_value=18, value=12, step=1)

    if st.button("执行家庭风险评级", type="primary", key="btn_family_risk"):
        result = assess_family_risk(
            minor_mode_enabled=minor_mode,
            historical_claims=hist_claims,
            child_age=child_age
        )
        st.session_state["family_risk"] = result

    if "family_risk" in st.session_state:
        r = st.session_state["family_risk"]
        # 评级展示
        color = {"low": "🟢", "medium": "🟡", "high": "🔴"}[r["risk_level"]]
        st.success(color + " 风险评级：" + r["risk_level_text"] + "（风险分" + str(r["risk_score"]) + "）")
        st.write("评级依据：")
        for reason in r["reasons"]:
            st.write("  - " + reason)

        # 定价方案
        plan = calculate_premium("family", r["risk_level"])
        st.divider()
        st.subheader("承保定价方案")
        st.markdown("**" + plan["plan_name"] + "**")
        m1, m2, m3 = st.columns(3)
        m1.metric("年保费", str(plan["annual_premium"]) + "元")
        m2.metric("免赔额", str(plan["deductible"]) + "元")
        m3.metric("保额", str(plan["coverage"]) + "元")

        # 三档对比
        st.subheader("三档价格对比")
        import pandas as pd
        compare_data = []
        for lvl, lvl_text in [("low", "低风险"), ("medium", "中风险"), ("high", "高风险")]:
            p = calculate_premium("family", lvl)
            compare_data.append({
                "风险等级": lvl_text,
                "年保费（元）": p["annual_premium"],
                "免赔额（元）": p["deductible"],
                "保额（元）": p["coverage"]
            })
        df_compare = pd.DataFrame(compare_data)
        st.dataframe(df_compare, width="stretch", hide_index=True)

# ===================== B端主播风险评级 =====================
else:
    st.subheader("主播风险因素录入")
    col1, col2, col3 = st.columns(3)
    with col1:
        induce_score = st.slider("直播间诱导话术风险分（0-1）", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with col2:
        complaint_cnt = st.number_input("历史被投诉次数", min_value=0, max_value=20, value=0, step=1)
    with col3:
        minor_ratio = st.slider("粉丝中未成年人占比", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    if st.button("执行主播风险评级", type="primary", key="btn_anchor_risk"):
        result = assess_anchor_risk(
            induce_risk_score=induce_score,
            complaint_count=complaint_cnt,
            minor_fan_ratio=minor_ratio
        )
        st.session_state["anchor_risk"] = result

    if "anchor_risk" in st.session_state:
        r = st.session_state["anchor_risk"]
        color = {"low": "🟢", "medium": "🟡", "high": "🔴"}[r["risk_level"]]
        st.success(color + " 风险评级：" + r["risk_level_text"] + "（风险分" + str(r["risk_score"]) + "）")
        st.write("评级依据：")
        for reason in r["reasons"]:
            st.write("  - " + reason)

        plan = calculate_premium("anchor", r["risk_level"])
        st.divider()
        st.subheader("承保定价方案")
        st.markdown("**" + plan["plan_name"] + "**")
        m1, m2, m3 = st.columns(3)
        m1.metric("年保费", str(plan["annual_premium"]) + "元")
        m2.metric("免赔额", str(plan["deductible"]) + "元")
        m3.metric("保额", str(plan["coverage"]) + "元")

        st.subheader("三档价格对比")
        import pandas as pd
        compare_data = []
        for lvl, lvl_text in [("low", "低风险"), ("medium", "中风险"), ("high", "高风险")]:
            p = calculate_premium("anchor", lvl)
            compare_data.append({
                "风险等级": lvl_text,
                "年保费（元）": p["annual_premium"],
                "免赔额（元）": p["deductible"],
                "保额（元）": p["coverage"]
            })
        df_compare = pd.DataFrame(compare_data)
        st.dataframe(df_compare, width="stretch", hide_index=True)

st.divider()
st.info("说明：MVP阶段风险因素为手动录入演示。后续接入风控模型输出——家庭页面读取session中的用户风险结果，主播页面读取直播间诱导话术检测分，实现自动评级定价。")
