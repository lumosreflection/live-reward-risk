import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

st.set_page_config(page_title="风控系统全局概览", layout="wide")

# ===================== 页面美化样式 =====================
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8ebf2;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 3px 10px rgba(31, 45, 80, 0.06);
}
[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 700;}
[data-testid="stMetricLabel"] {font-size: 0.92rem; color: #5b6478;}
.dashboard-title {
    background: linear-gradient(90deg, #1f3a68 0%, #2e6aa8 100%);
    padding: 22px 28px; border-radius: 16px; margin-bottom: 8px;
}
.dashboard-title h1 {color: #ffffff; margin: 0; font-size: 1.7rem;}
.dashboard-title p {color: #d6e4f5; margin: 6px 0 0 0; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-title">
<h1>未成年直播打赏智能风控系统 · 全局概览</h1>
<p>实时汇总平台用户风险态势、话术识别结果与理赔业务情况</p>
</div>
""", unsafe_allow_html=True)

# ===================== 读取真实数据 =====================
result_df = st.session_state.get("result_df", None)

flag_cols = ["risk_fast_charge", "risk_single_large", "risk_spike",
             "risk_silent_burst", "risk_night_large"]
type_name_map = {
    "risk_fast_charge": "短时高频",
    "risk_single_large": "单笔大额",
    "risk_spike": "消费暴涨",
    "risk_silent_burst": "沉寂爆发",
    "risk_night_large": "夜间大额"
}

if result_df is None:
    st.warning("尚未检测到风控数据。")
    st.markdown("请先到左侧【平台风控后台】：**上传直播打赏csv文件后，运行用户行为异常检测模块**，再回到本概览栏查看全局态势。")
    st.stop()

# 风险等级判定
def user_level(row):
    if row["is_high_risk"] == 1:
        return "高风险"
    if any(str(row[c]) == "存在风险" for c in flag_cols):
        return "中风险"
    return "正常"

result_df = result_df.copy()
result_df["风险等级"] = result_df.apply(user_level, axis=1)

total_users = len(result_df)
n_high = (result_df["风险等级"] == "高风险").sum()
n_mid = (result_df["风险等级"] == "中风险").sum()
n_normal = total_users - n_high - n_mid
high_ratio = n_high / total_users if total_users > 0 else 0
total_money = result_df["total_gift_money"].sum()

# 理赔数据（可能未初始化）
claim_cases = st.session_state.get("claim_cases", None)
pending_claims = None
if claim_cases is not None:
    pending_claims = len([c for c in claim_cases if c["status"] == "待审核"])

# ===================== 数字卡片 第一行 =====================
st.markdown("#### 用户风险总览")
c1, c2, c3 = st.columns(3)
c1.metric("检测用户总数", str(total_users) + " 人")
c2.metric("高风险用户", str(n_high) + " 人")
c3.metric("中风险用户", str(n_mid) + " 人")

# ===================== 数字卡片 第二行 =====================
c4, c5, c6 = st.columns(3)
c4.metric("高风险用户占比", str(round(high_ratio * 100, 1)) + " %")
if pending_claims is not None:
    c5.metric("待审核理赔案件", str(pending_claims) + " 件")
else:
    c5.metric("待审核理赔案件", "—")
    st.caption("提示：进入左侧「理赔审核工作台」页面后，此处将显示待审核案件数")
c6.metric("累计涉及打赏金额", str(total_money) + " 元")

st.divider()

# ===================== 图表 第一行：等级分布环形图 + 风险类型柱状图 =====================
col_fig1, col_fig2 = st.columns(2)

with col_fig1:
    st.markdown("#### 用户风险等级分布")
    fig1, ax1 = plt.subplots(figsize=(6.2, 5))
    pie_vals = [n_high, n_mid, n_normal]
    pie_labels = ["高风险", "中风险", "正常"]
    pie_colors = ["#E74C3C", "#F39C12", "#27AE60"]
    # 过滤0值
    pv, pl, pc = [], [], []
    for v, l, c in zip(pie_vals, pie_labels, pie_colors):
        if v > 0:
            pv.append(v); pl.append(l); pc.append(c)
    wedges, texts, autotexts = ax1.pie(
        pv, labels=pl, colors=pc, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
        pctdistance=0.78, textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax1.text(0, 0.08, str(total_users), ha="center", va="center",
             fontsize=26, fontweight="bold", color="#2c3e50")
    ax1.text(0, -0.15, "用户总数", ha="center", va="center",
             fontsize=11, color="#7f8c9b")
    st.pyplot(fig1)

with col_fig2:
    st.markdown("#### 风险类型命中用户数")
    type_counts = []
    for col in flag_cols:
        type_counts.append((type_name_map[col], (result_df[col] == "存在风险").sum()))
    type_counts.sort(key=lambda x: x[1])
    names = [x[0] for x in type_counts]
    vals = [x[1] for x in type_counts]
    fig2, ax2 = plt.subplots(figsize=(6.2, 5))
    bars = ax2.barh(names, vals, color="#2E86AB", height=0.6)
    # 渐变色
    cmap = plt.cm.Blues
    maxv = max(vals) if max(vals) > 0 else 1
    for i, bar in enumerate(bars):
        bar.set_color(cmap(0.45 + 0.45 * vals[i] / maxv))
    for i, v in enumerate(vals):
        ax2.text(v + maxv * 0.02, i, str(v), va="center",
                 fontsize=11, color="#2c3e50", fontweight="bold")
    ax2.set_xlabel("命中用户数")
    ax2.set_xlim(0, maxv * 1.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)
    st.pyplot(fig2)

st.divider()

# ===================== 图表 第二行：高风险用户打赏金额 Top =====================
st.markdown("#### 高风险用户累计打赏金额 TOP 10")
top_df = result_df[result_df["风险等级"] == "高风险"].copy()
top_df = top_df.sort_values("total_gift_money", ascending=False).head(10)

if len(top_df) > 0:
    fig3, ax3 = plt.subplots(figsize=(11, 4.6))
    labels = ["****" + str(u)[-4:] for u in top_df["user_id"]]
    money = top_df["total_gift_money"].tolist()
    bars = ax3.bar(labels, money, color="#E74C3C", width=0.62)
    for bar, m in zip(bars, money):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(m), ha="center", va="bottom", fontsize=10,
                 color="#c0392b", fontweight="bold")
    ax3.set_ylabel("累计打赏金额（元）")
    ax3.set_xlabel("用户（ID已脱敏）")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20)
    st.pyplot(fig3)
else:
    st.info("当前无高风险用户。")

st.divider()
st.caption("数据来源：平台风控后台实时检测结果。本页面仅读取展示，不修改任何数据。")#（用户行为模型 + 关键词话术识别）
