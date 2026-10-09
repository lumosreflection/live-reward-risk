# -*- coding: utf-8 -*-
"""
merge_danmu_dataset.py  (v2 增加livechat_clean.csv)
功能：合并【队友正样本】+【三个负样本来源】→ 均衡 → 清洗 → 划分 train/val/test
负样本来源：
  1. 队友直播话术.csv 自带的 label=0
  2. danmu_neg_sim.txt（仿真正常弹幕）
  3. livechat_clean.csv（真实直播正常弹幕）
运行：conda activate minor_risk && python merge_danmu_dataset.py
"""
import pandas as pd
import os

# ==================== 配置区（按实际路径改） ====================
DATASET_DIR = "dataset"
POS_CSV      = os.path.join(DATASET_DIR, "直播话术.csv")       # 队友正样本
NEG_TXT      = os.path.join(DATASET_DIR, "danmu_neg_sim.txt")  # 仿真负样本txt
NEG_LIVE     = os.path.join(DATASET_DIR, "livechat_clean.csv") # 真实直播负样本csv
NEG_RATIO    = 1.2     # 负样本:正样本 比例
SEED         = 42
# ==============================================================

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------- 1. 队友csv（正样本为主） ----------
print("=" * 60)
print("[1] 读取队友话术csv ...")
df_team = pd.read_csv(POS_CSV, encoding="utf-8-sig")
df_team = df_team.rename(columns={df_team.columns[0]: "text", df_team.columns[1]: "label"})
pos_df = df_team[df_team["label"] == 1][["text"]].copy()
pos_df["label"] = 1
team_neg_df = df_team[df_team["label"] == 0][["text"]].copy()
team_neg_df["label"] = 0
print(f"  正样本: {len(pos_df)}，队友自带负样本: {len(team_neg_df)}")

# ---------- 2. 仿真txt负样本 ----------
print("[2] 读取仿真负样本txt ...")
txt_neg_list = []
if os.path.exists(NEG_TXT):
    with open(NEG_TXT, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                txt_neg_list.append({"text": s, "label": 0})
    print(f"  txt负样本: {len(txt_neg_list)}")
else:
    print(f"  ⚠ 未找到 {NEG_TXT}，跳过")

# ---------- 3. 真实直播负样本csv ----------
print("[3] 读取livechat_clean.csv ...")
live_neg_list = []
if os.path.exists(NEG_LIVE):
    live_df = pd.read_csv(NEG_LIVE, encoding="utf-8-sig")
    live_df = live_df.rename(columns={live_df.columns[0]: "text", live_df.columns[1]: "label"})
    live_neg_df = live_df[live_df["label"] == 0][["text"]].copy()
    live_neg_df["label"] = 0
    print(f"  livechat负样本: {len(live_neg_df)}")
else:
    live_neg_df = pd.DataFrame(columns=["text", "label"])
    print(f"  ⚠ 未找到 {NEG_LIVE}，跳过")

# ---------- 4. 合并全部负样本 ----------
print("[4] 合并全部负样本 ...")
all_neg_raw = pd.concat([team_neg_df, pd.DataFrame(txt_neg_list), live_neg_df], ignore_index=True)
# 去重
all_neg_raw = all_neg_raw.drop_duplicates(subset="text").reset_index(drop=True)
print(f"  合并去重后负样本池: {len(all_neg_raw)}")

# 均衡抽样
n_pos = len(pos_df)
need_neg = int(n_pos * NEG_RATIO)
if len(all_neg_raw) >= need_neg:
    neg_final = all_neg_raw.sample(need_neg, random_state=SEED).reset_index(drop=True)
    print(f"  均衡抽样取负样本 {need_neg} 条（比例{NEG_RATIO}:1）")
else:
    neg_final = all_neg_raw
    print(f"  ⚠ 负样本不足，全部使用 {len(neg_final)} 条")

# ---------- 5. 合并正负 + 打乱 ----------
full_df = pd.concat([pos_df, neg_final], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"  合并后总样本: {len(full_df)}  正: {(full_df['label']==1).sum()}  负: {(full_df['label']==0).sum()}")

# ---------- 6. 划分 8:1:1 ----------
total = len(full_df)
train_end = int(total * 0.8)
val_end = int(total * 0.9)
train_df = full_df.iloc[:train_end].reset_index(drop=True)
val_df   = full_df.iloc[train_end:val_end].reset_index(drop=True)
test_df  = full_df.iloc[val_end:].reset_index(drop=True)

# ---------- 7. 保存 ----------
train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False, encoding="utf-8-sig")
val_df.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False, encoding="utf-8-sig")
test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False, encoding="utf-8-sig")
full_df.to_csv(os.path.join(DATASET_DIR, "speech_data.csv"), index=False, encoding="utf-8-sig")

print("=" * 60)
print(" 保存完成（dataset/ 下）：")
print(f"  train.csv         {len(train_df)} 条（正{(train_df['label']==1).sum()}/负{(train_df['label']==0).sum()}）")
print(f"  val.csv           {len(val_df)} 条（正{(val_df['label']==1).sum()}/负{(val_df['label']==0).sum()}）")
print(f"  test.csv          {len(test_df)} 条（正{(test_df['label']==1).sum()}/负{(test_df['label']==0).sum()}）")
print(f"  speech_data.csv   {len(full_df)} 条（全量，给Colab微调用）")
print("=" * 60)
