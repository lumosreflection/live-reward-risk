import streamlit as st
import pandas as pd
from user_risk_func import process_claim, detect_identity_spoofing, classify_review_grade

st.set_page_config(page_title="理赔审核工作台", layout="wide")
st.title("理赔审核工作台")
st.markdown("保险公司视角：案件受理 → AI责任划分 → 赔款计算 → 审核结论")

# 切换案件时清空AI审核结果，回到未审核状态
def clear_ai_result():
    st.session_state.ai_result = None

# ===================== 初始化mock案件数据 =====================
# ===================== 审核员最终操作留痕记录 =====================
if "review_audit_log" not in st.session_state:
    st.session_state.review_audit_log = []

if "claim_cases" not in st.session_state:
    st.session_state.claim_cases = [
        {
            "case_id": "CLM-2026-001",
            "applicant": "张某某（监护人）",
            "applicant_type": "C端家庭",
            "loss_amount": 3200,
            "platform_refund": 0,
            "family_risk_level": "medium",
            "has_anchor_induce": True,
            "induce_level": "severe",
            "has_guardian_neglect": False,
            "neglect_level": "none",
            "annual_paid": 0,
            "status": "待审核",
            "identity_evidence": {
                "night_ratio": 0.35,
                "school_ratio": 0.20,
                "delete_record": 0,
                "habit_shift": 0,
                "face_verify_fail": 0
            },
            "case_scene": "监护人自主申请",
            "account_user_id": "u_1001",
            "evidence_summary": "打赏流水3200元；音频转写检测到'刷礼物冲榜'等诱导话术5句；主播风险等级高",
            "final_payout": None,
            "verdict": None
        },
        {
            "case_id": "CLM-2026-002",
            "applicant": "李某某（监护人）",
            "applicant_type": "C端家庭",
            "loss_amount": 1500,
            "platform_refund": 0,
            "family_risk_level": "low",
            "has_anchor_induce": False,
            "induce_level": "none",
            "has_guardian_neglect": False,
            "neglect_level": "none",
            "annual_paid": 0,
            "status": "待审核",
            "identity_evidence": {
                "night_ratio": 0.05,
                "school_ratio": 0.05,
                "delete_record": 0,
                "habit_shift": 0,
                "face_verify_fail": 0
            },
            "case_scene": "平台巡检发现",
            "account_user_id": "u_1002",
            "evidence_summary": "打赏流水1500元；音频转写未检测到诱导话术；监护人已开启未成年人模式",
            "final_payout": None,
            "verdict": None
        },
        {
            "case_id": "CLM-2026-003",
            "applicant": "王某某（监护人）",
            "applicant_type": "C端家庭",
            "loss_amount": 5000,
            "platform_refund": 1000,
            "family_risk_level": "high",
            "has_anchor_induce": False,
            "induce_level": "none",
            "has_guardian_neglect": True,
            "neglect_level": "severe",
            "annual_paid": 0,
            "status": "待审核",
            "identity_evidence": {
                "night_ratio": 0.55,
                "school_ratio": 0.40,
                "delete_record": 2,
                "habit_shift": 1,
                "face_verify_fail": 2
            },
            "case_scene": "监管转办",
            "account_user_id": "u_1003",
            "evidence_summary": "打赏流水5000元；平台已退款1000元；监护人未开启未成年人模式，设备长期放任使用",
            "final_payout": None,
            "verdict": None
        },
        {
            "case_id": "CLM-2026-004",
            "applicant": "某某MCN公会",
            "applicant_type": "B端公会",
            "loss_amount": 8000,
            "platform_refund": 0,
            "family_risk_level": "medium",
            "has_anchor_induce": True,
            "induce_level": "mild",
            "has_guardian_neglect": True,
            "neglect_level": "mild",
            "annual_paid": 0,
            "status": "待审核",
            "identity_evidence": {
                "night_ratio": 0.25,
                "school_ratio": 0.10,
                "delete_record": 0,
                "habit_shift": 0,
                "face_verify_fail": 0
            },
            "case_scene": "匿名举报",
            "account_user_id": "u_1004",
            "evidence_summary": "主播被索赔8000元；直播间存在轻微诱导话术；监护人存在一般失职",
            "final_payout": None,
            "verdict": None
        }
    ]
if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "closed_claims" not in st.session_state:
    st.session_state.closed_claims = []
    st.session_state.ai_result = None

# ===================== 顶部统计卡片 =====================
cases = st.session_state.claim_cases
pending = len([c for c in cases if c["status"] == "待审核"])
approved = len([c for c in cases if c["status"] == "已通过"])
rejected = len([c for c in cases if c["status"] == "已拒赔"])
total_loss = sum([c["loss_amount"] for c in cases])

c1, c2, c3, c4 = st.columns(4)
c1.metric("待审核案件", pending)
c2.metric("已通过", approved)
c3.metric("已拒赔", rejected)
c4.metric("涉及总金额", str(total_loss) + "元")

st.divider()

# ===================== 案件列表 =====================
st.subheader("理赔案件列表")
df_cases = pd.DataFrame([
    {
        "案件编号": c["case_id"],
        "申请人": c["applicant"],
        "类型": c["applicant_type"],
        "损失金额": str(c["loss_amount"]) + "元",
        "平台退款": str(c["platform_refund"]) + "元",
        "状态": c["status"],
        "最终赔付": str(c["final_payout"]) + "元" if c["final_payout"] is not None else "—",
        "冒用风险": detect_identity_spoofing(c.get("identity_evidence", {}))["spoofing_level"],
        "案件来源": c.get("case_scene", "—")
    }
    for c in cases
])
st.dataframe(df_cases, width="stretch", hide_index=True)

# 选择案件
case_ids = [c["case_id"] for c in cases]
selected_id = st.selectbox("选择要审核的案件编号", case_ids, key="case_selector", on_change=clear_ai_result)
st.session_state.selected_case_id = selected_id
selected_case = next(c for c in cases if c["case_id"] == selected_id)

st.divider()

# ===================== 案件详情 =====================
st.subheader("案件详情：" + selected_case["case_id"])

col_info, col_evidence, col_factor = st.columns(3)

with col_info:
    st.markdown("**基本信息**")
    st.write("申请人：" + selected_case["applicant"])
    st.write("申请类型：" + selected_case["applicant_type"])
    st.write("损失金额：" + str(selected_case["loss_amount"]) + "元")
    st.write("平台已退款：" + str(selected_case["platform_refund"]) + "元")
    st.write("家庭风险等级：" + selected_case["family_risk_level"])
    st.write("本年度已赔付：" + str(selected_case["annual_paid"]) + "元")
    st.write("当前状态：" + selected_case["status"])
    st.write("案件来源：" + selected_case.get("case_scene", "—"))

with col_evidence:
    st.markdown("**证据摘要**")
    st.info(selected_case["evidence_summary"])

with col_factor:
    st.markdown("**责任判定要素**")
    induce_text = {"none": "无", "mild": "轻微诱导", "severe": "刻意诱导"}[selected_case["induce_level"]]
    neglect_text = {"none": "尽责", "mild": "一般失职", "severe": "严重失职"}[selected_case["neglect_level"]]
    st.write("主播诱导：" + induce_text)
    st.write("监护人失职：" + neglect_text)
    st.write("平台退款：" + str(selected_case["platform_refund"]) + "元")

st.divider()

# ===================== 身份冒用核验（任务2新增） =====================
st.subheader("身份冒用核验")
id_ev = selected_case.get("identity_evidence", {})
i1, i2, i3, i4, i5 = st.columns(5)
i1.metric("凌晨打赏占比", str(round(id_ev.get("night_ratio", 0) * 100, 1)) + "%")
i2.metric("上课时段占比", str(round(id_ev.get("school_ratio", 0) * 100, 1)) + "%")
i3.metric("删除记录次数", str(id_ev.get("delete_record", 0)) + "次")
i4.metric("账号作息突变", "是" if id_ev.get("habit_shift", 0) == 1 else "否")
i5.metric("人脸核验失败", str(id_ev.get("face_verify_fail", 0)) + "次")

if st.button("执行身份冒用核验", key="btn_spoofing"):
    spoof = detect_identity_spoofing(id_ev)
    st.session_state.spoof_result = spoof
    if spoof["is_manual_review"]:
        st.error("核验结论：高风险冒用，需人工重点核验")
    elif spoof["spoofing_level"] == "可疑":
        st.warning("核验结论：可疑，建议人工复核")
    else:
        st.success("核验结论：正常")
    st.write("冒用风险评分：" + str(spoof["spoofing_score"]) + " / 100")
    if spoof["reasons"]:
        st.markdown("**风险原因：**")
        for r in spoof["reasons"]:
            st.write("  - " + r)
    else:
        st.info("未发现身份冒用特征")

st.divider()

# ===================== AI一键审核 =====================
st.subheader("AI一键审核：责任划分 + 赔款计算")

if st.button("执行AI一键审核", key="btn_ai_review", type="primary"):
    with st.spinner("AI风控核验中..."):
        st.session_state.review_grade = classify_review_grade(selected_case, st.session_state.get("spoof_result"))
        result = process_claim(
            loss_amount=selected_case["loss_amount"],
            family_risk_level=selected_case["family_risk_level"],
            has_anchor_induce=selected_case["has_anchor_induce"],
            induce_level=selected_case["induce_level"],
            has_guardian_neglect=selected_case["has_guardian_neglect"],
            neglect_level=selected_case["neglect_level"],
            platform_refund=selected_case["platform_refund"],
            annual_paid=selected_case["annual_paid"]
        )
    st.session_state.ai_result = result

# 展示AI审核结果
if st.session_state.ai_result is not None:
    res = st.session_state.ai_result

    # 赔付结论
    if res["is_approved"]:
        st.success("AI建议：审核通过，赔付 " + str(res["final_payout"]) + " 元")
    else:
        st.error("AI建议：不予赔付")

    st.markdown("**判定结论：** " + res["final_verdict"])

    # 审核分级
    st.markdown("**审核分级：**")
    grade = st.session_state.get("review_grade", {})
    st.write("分级：" + grade.get("grade", "-") + "级 " + grade.get("grade_name", "") + "，建议处理时限 " + str(grade.get("audit_hours", "-")) + " 小时")
    for r in grade.get("rules", []):
        st.write("  - " + r)

    # 责任划分
    st.markdown("**责任划分：**")
    resp = res["responsibility"]
    st.write("责任方：" + "、".join(resp["responsibility_parties"]))
    st.write("保险赔付比例：" + str(round(resp["insurance_pay_ratio"] * 100, 1)) + "%")
    for d in resp["detail"]:
        st.write("  - " + d)

    # 赔款计算
    st.markdown("**赔款计算明细：**")
    comp = res["compensation"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("免赔额", str(comp["deductible"]) + "元")
    m2.metric("扣退款后", str(comp["after_refund"]) + "元")
    m3.metric("扣免赔后", str(comp["after_deductible"]) + "元")
    m4.metric("年度剩余", str(comp["annual_remaining"]) + "元")
    for d in comp["detail"]:
        st.write("  - " + d)

    st.divider()

    # 审核员最终操作
    st.caption("审核员最终操作")
    col_pass, col_reject = st.columns(2)
    with col_pass:
        approve_payout = res["final_payout"]
        approve_label = "审核通过（按AI建议赔付）" if approve_payout > 0 else "审核通过（按AI建议不赔付）"
        if st.button(approve_label, key="btn_pass"):
            approve_action = "审核通过（赔付）" if approve_payout > 0 else "审核通过（不赔付）"
            close_result = "赔付" if approve_payout > 0 else "不赔付"
            st.session_state.review_audit_log.append({
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "case_id": selected_case["case_id"],
                "account_user_id": selected_case.get("account_user_id", "未知账号"),
                "ai_verdict": res["final_verdict"],
                "reviewer_action": approve_action,
                "final_payout": approve_payout
            })
            for c in st.session_state.claim_cases:
                if c["case_id"] == selected_case["case_id"]:
                    c["status"] = "已通过"
                    c["final_payout"] = approve_payout
                    c["verdict"] = res["final_verdict"]
            st.session_state.closed_claims.append({
                "case_id": selected_case["case_id"],
                "account_user_id": selected_case.get("account_user_id", "未知账号"),
                "close_time": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "close_result": close_result
            })
            if approve_payout > 0:
                st.success("案件 " + selected_case["case_id"] + " 已审核通过，赔付 " + str(approve_payout) + "元")
            else:
                st.success("案件 " + selected_case["case_id"] + " 已审核通过（AI建议不赔付，按建议执行）")
            st.rerun()
    with col_reject:
        if st.button("拒赔", key="btn_reject"):
            st.session_state.review_audit_log.append({
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "case_id": selected_case["case_id"],
                "account_user_id": selected_case.get("account_user_id", "未知账号"),
                "ai_verdict": res["final_verdict"],
                "reviewer_action": "拒赔",
                "final_payout": 0
            })
            for c in st.session_state.claim_cases:
                if c["case_id"] == selected_case["case_id"]:
                    c["status"] = "已拒赔"
                    c["final_payout"] = 0
                    c["verdict"] = res["final_verdict"]
            st.session_state.closed_claims.append({
                "case_id": selected_case["case_id"],
                "account_user_id": selected_case.get("account_user_id", "未知账号"),
                "close_time": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "close_result": "拒赔"
            })
            st.warning("案件 " + selected_case["case_id"] + " 已拒赔")
            st.rerun()

    # 审核操作记录（按当前案件过滤，跟随案件走）
    cur_review_logs = [x for x in st.session_state.review_audit_log if x["case_id"] == selected_case["case_id"]]
    if len(cur_review_logs) > 0:
        st.divider()
        st.caption("该案件审核操作记录")
        st.dataframe(pd.DataFrame(cur_review_logs), width="stretch", hide_index=True)
else:
    st.info("点击上方【执行AI一键审核】按钮，自动完成责任划分与赔款计算")
