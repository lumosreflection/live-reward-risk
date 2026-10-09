import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
# ========== 修复matplotlib中文方框：注册项目自带开源中文字体（云端Linux无中文字体） ==========
from user_risk_func import setup_cn_font
setup_cn_font()
import matplotlib.pyplot as plt

from user_risk_func import long_text_speech_risk_detect_rule

st.title("主播直播间风险预警后台")
st.markdown("仅展示本直播间脱敏风险信息，保护用户隐私；不可查看用户完整个人信息。")
st.divider()

# ===================== 读取真实检测结果 =====================
result_df = st.session_state.get("result_df", None)
raw_df = st.session_state.get("df_data", None)

if result_df is None or raw_df is None:
    st.warning("尚未检测到风控数据。")
    st.markdown("请先到左侧【平台风控后台】：**上传送礼csv → 运行用户行为异常检测**，再回到本页面。")
    st.stop()

# ===================== 建立 直播间-观众 关联，并带上用户风险结果 =====================
raw_df = raw_df.copy()
raw_df["gift_time"] = pd.to_datetime(raw_df["gift_time"])
relation = raw_df[["room_id", "user_id"]].drop_duplicates()
relation_risk = relation.merge(result_df, on="user_id", how="left")

# ===================== 直播间选择 =====================
st.subheader("选择直播间")
room_list = sorted(raw_df["room_id"].unique().tolist())
selected_room = st.selectbox("切换直播间", room_list, key="anchor_room_select")

# 该直播间的观众风险数据
room_risk = relation_risk[relation_risk["room_id"] == selected_room].copy()
room_raw = raw_df[raw_df["room_id"] == selected_room].copy()

# 风险等级判定
flag_cols = ["risk_fast_charge", "risk_single_large", "risk_spike",
             "risk_silent_burst", "risk_night_large"]

def audience_level(row):
    if row["is_high_risk"] == 1:
        return "🔴高风险"
    if any(str(row[c]) == "存在风险" for c in flag_cols):
        return "🟡中风险"
    return "🟢正常"

room_risk["观众风险等级"] = room_risk.apply(audience_level, axis=1)

# ===================== 统计卡片（真实） =====================
n_total = len(room_risk)
n_high = (room_risk["观众风险等级"] == "🔴高风险").sum()
n_mid = (room_risk["观众风险等级"] == "🟡中风险").sum()
room_money = room_raw["gift_value"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("本直播间观众数", n_total)
c2.metric("高风险观众", n_high)
c3.metric("中风险观众", n_mid)
c4.metric("本直播间打赏总额", str(room_money) + "元")

if n_high > 0:
    st.warning("系统提示：本直播间检测到" + str(n_high) + "名高风险观众，请规范话术，避免诱导非理性打赏。")

st.divider()

# ===================== 风险类型分布饼图（真实统计） =====================
st.subheader("本直播间观众风险类型分布")
type_map = {
    "risk_fast_charge": "短时高频",
    "risk_single_large": "单笔大额",
    "risk_spike": "消费暴涨",
    "risk_silent_burst": "沉寂爆发",
    "risk_night_large": "夜间大额"
}
pie_labels = []
pie_values = []
for col, name in type_map.items():
    cnt = (room_risk[col] == "存在风险").sum()
    if cnt > 0:
        pie_labels.append(name)
        pie_values.append(cnt)

if len(pie_labels) > 0:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C", "#4DABF7"]
    ax.pie(pie_values, labels=pie_labels, autopct="%1.0f%%",
           startangle=90, colors=colors[:len(pie_labels)])
    ax.set_title("观众风险类型人数分布")
    st.pyplot(fig)
else:
    st.info("本直播间观众未命中任何风险类型。")

st.divider()

# ===================== 脱敏观众名单（真实） =====================
st.subheader(" 脱敏观众名单-后4位显示")

def mask_uid(uid):
    s = str(uid)
    return "****" + s[-4:] if len(s) >= 4 else "****"

def main_tags(row):
    tags = [name for col, name in type_map.items() if str(row[col]) == "存在风险"]
    return "、".join(tags) if tags else "无"

mask_df = pd.DataFrame({
    "用户ID(脱敏)": room_risk["user_id"].apply(mask_uid),
    "风险等级": room_risk["观众风险等级"],
    "主要风险标签": room_risk.apply(main_tags, axis=1),
    "累计打赏(元)": room_risk["total_gift_money"]
})
# 高风险排前面
sort_order = {"🔴高风险": 0, "🟡中风险": 1, "🟢正常": 2}
mask_df["_sort"] = mask_df["风险等级"].map(sort_order)
mask_df = mask_df.sort_values("_sort").drop(columns="_sort")
st.dataframe(mask_df, width="stretch", hide_index=True)

st.divider()


# ===================== 实时预警与待办（整合建议4新增） =====================
st.subheader("实时预警与待办")
if n_high > 0:
    st.error("系统待办：本直播间有 " + str(n_high) + " 名高风险观众待处理")
    high_short = pd.DataFrame({
        "高风险观众(脱敏)": room_risk[room_risk["观众风险等级"] == "高风险"]["user_id"].apply(mask_uid),
        "主要风险标签": room_risk[room_risk["观众风险等级"] == "高风险"].apply(main_tags, axis=1),
        "累计打赏(元)": room_risk[room_risk["观众风险等级"] == "高风险"]["total_gift_money"]
    })
    st.dataframe(high_short, width="stretch", hide_index=True)

st.markdown("**主播待办事项**")
t1 = st.checkbox("对高风险观众账号发起重点关注标记", key="anchor_todo_focus")
t2 = st.checkbox("复核本场直播话术，规避诱导用语", key="anchor_todo_speech")
t3 = st.checkbox("必要时上报平台风控或联系监护人", key="anchor_todo_report")
done_n = sum([t1, t2, t3])
if done_n == 3:
    st.success("待办已全部处理，本直播间预警闭环")
elif done_n > 0:
    st.info("已处理 " + str(done_n) + " / 3 项待办")
else:
    st.warning("存在未处理待办，请优先处理高风险观众")

# ===================== 本场直播话术风险检测（复用关键词检测） =====================
st.subheader("本场直播话术风险检测")
default_text = "欢迎大家来到直播间，今天正常聊天。喜欢主播的可以支持一下，点点关注就行，不用刷礼物。"
speech_text = st.text_area("本场直播话术（可由音频自动转写生成，也可手动粘贴）",
                           value=default_text, height=120, key="anchor_speech_text")
if st.button("检测话术风险", key="anchor_btn_speech"):
    r = long_text_speech_risk_detect_rule(speech_text)
    if r["whole_risk"]:
        st.error("检测到 " + str(len(r["risk_sentences"])) + " 句诱导风险话术，请立即规范用语！")
        for s in r["risk_sentences"]:
            st.write("  ⚠" + s["sentence"])
    else:
        st.success("未检测到诱导风险话术，用语规范。")
