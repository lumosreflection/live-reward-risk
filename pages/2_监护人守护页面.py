import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from user_risk_func import assess_consumption_rationality

st.title("监护人‑未成年账号守护中心")
st.markdown("查看绑定账号的真实风控结果与消费记录，配置消费防护策略。")
st.divider()

# ===================== 读取平台风控后台的真实检测结果 =====================
result_df = st.session_state.get("result_df", None)
raw_df = st.session_state.get("df_data", None)

if result_df is None or raw_df is None:
    st.warning("尚未检测到风控数据。")
    st.markdown("请先到左侧【平台风控后台】页面：**上传送礼csv → 点击「运行用户行为异常检测」**，然后再回到本页面。")
    st.info("本页面数据来自平台风控后台的实时检测结果，不使用写死数据。")
    st.stop()

# ===================== 账号选择（模拟家长已绑定的孩子账号） =====================
st.subheader("绑定的未成年账号")
user_list = sorted(result_df["user_id"].unique().tolist())
default_idx = 0
jump_user = st.session_state.get("guard_jump_user", None)
if jump_user in user_list:
    default_idx = user_list.index(jump_user)
selected = st.selectbox("选择要查看的账号", user_list, key="guard_user_select", index=default_idx)

# 该用户的风控结果行
user_row = result_df[result_df["user_id"] == selected].iloc[0]

# 综合风险等级判定
flag_cols = ["risk_fast_charge", "risk_single_large", "risk_spike",
             "risk_silent_burst", "risk_night_large"]
has_any_flag = any(str(user_row[c]) == "存在风险" for c in flag_cols)
if user_row["is_high_risk"] == 1:
    level_text = "🔴 高风险"
elif has_any_flag:
    level_text = "🟡 中风险"
else:
    level_text = "🟢 低风险"

col1, col2, col3 = st.columns(3)
col1.metric("账号ID", selected)
col2.metric("登记年龄（模拟）", "14岁")
col3.metric("综合风险等级", level_text)

st.divider()

# ===================== 消费总览（真实数据） =====================
st.subheader("近期消费总览")
user_raw = raw_df[raw_df["user_id"] == selected].copy()
user_raw["gift_time"] = pd.to_datetime(user_raw["gift_time"])
total = user_raw["gift_value"].sum()
cnt = len(user_raw)
rooms = user_raw["room_id"].nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("累计总打赏", str(total) + "元")
m2.metric("打赏笔数", str(cnt) + "笔")
m3.metric("涉及直播间", str(rooms) + "个")
m4.metric("夜间打赏金额", str(user_row["night_gift_money"]) + "元")

st.divider()

# ===================== 消费理性评估（任务1新增） =====================
st.subheader("消费理性评估")
rationality_df = assess_consumption_rationality(raw_df, result_df)
rationality_row = rationality_df[rationality_df["user_id"] == selected].iloc[0]
r_score = int(rationality_row["rationality_score"])
r_level = rationality_row["rationality_level"]

c1, c2 = st.columns([1, 1])
c1.metric("理性评分", str(r_score) + " / 100")
c2.metric("评分等级", r_level)
st.progress(r_score / 100)

r1, r2, r3 = st.columns(3)
r1.metric("上课时段打赏占比", str(round(rationality_row["school_time_ratio"] * 100, 1)) + "%")
r2.metric("小额多笔占比", str(round(rationality_row["small_multi_ratio"] * 100, 1)) + "%")
r3.metric("周末假期打赏占比", str(round(rationality_row["weekend_holiday_ratio"] * 100, 1)) + "%")

reasons = rationality_row["rationality_reasons"]
if reasons:
    st.markdown("**主要扣分原因**")
    for r in reasons:
        st.markdown("- " + r)
else:
    st.info("未发现明显非理性消费特征")

# ===================== 风险行为标签（真实检测结果） =====================
st.subheader("风险行为标签")
flag_df = pd.DataFrame([
    {"风险类型": "短时高频充值", "检测结果": user_row["risk_fast_charge"]},
    {"风险类型": "单笔大额消费", "检测结果": user_row["risk_single_large"]},
    {"风险类型": "消费突然暴涨", "检测结果": user_row["risk_spike"]},
    {"风险类型": "沉寂后集中充值", "检测结果": user_row["risk_silent_burst"]},
    {"风险类型": "夜间大额打赏", "检测结果": user_row["risk_night_large"]},
])
st.dataframe(flag_df, width="stretch", hide_index=True)

# 风险打赏流水明细（真实：全部打赏记录，标记夜间）
st.markdown("**打赏流水明细（夜间23:00‑05:00已标记）**")
detail = user_raw.sort_values("gift_time").copy()
detail["hour"] = detail["gift_time"].dt.hour
detail["是否夜间"] = detail["hour"].apply(lambda h: "夜间" if (h >= 23 or h <= 5) else "正常时段")
show_detail = detail[["gift_time", "room_id", "gift_value", "是否夜间"]].rename(
    columns={"gift_time": "打赏时间", "room_id": "直播间", "gift_value": "金额(元)"}
)
st.dataframe(show_detail, width="stretch", hide_index=True)

st.divider()

# ===================== 单人打赏时序图（复用平台页绘图逻辑） =====================
st.subheader("打赏时序曲线")
plot_df = user_raw.sort_values("gift_time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plot_df["gift_time"], plot_df["gift_value"], marker="o",
        linestyle="-", color="#2E86AB")
ax.set_xlabel("打赏时间")
ax.set_ylabel("单笔打赏金额")
ax.set_title("账号 " + selected + " 打赏时序曲线")
ax.grid(alpha=0.3)
plt.xticks(rotation=30)
st.pyplot(fig)

st.divider()

# ===================== 守护策略配置（模拟开关，不真实生效） =====================
st.subheader("守护策略配置")
sw1 = st.toggle("开启每日打赏限额", value=True, key="guard_sw1")
sw2 = st.toggle("夜间23:00‑05:00锁定充值", value=True, key="guard_sw2")
sw3 = st.toggle("大额消费推送消息提醒家长", value=True, key="guard_sw3")
st.info("以上为UI模拟，真实生效需对接平台账号接口；原型仅展示交互逻辑。")
