import os
# bert-base-chinese 已完整缓存，强制离线加载，避免联网超时
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 读取csv，这三个文件放在项目根目录
train_df = pd.read_csv("dataset/train.csv", encoding="utf-8-sig")
val_df = pd.read_csv("dataset/val.csv", encoding="utf-8-sig")
test_df = pd.read_csv("dataset/test.csv", encoding="utf-8-sig")

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

train_token = train_ds.map(tokenize_fn, batched=True)
val_token = val_ds.map(tokenize_fn, batched=True)
test_token = test_ds.map(tokenize_fn, batched=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 动态padding：每个batch内填充到该batch最长，解决句子长度不一无法组batch的问题
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

training_args = TrainingArguments(
    output_dir="./bert_live_speech_model",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    use_cpu=True   # 这里！替换旧的no_cuda
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=val_token,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 测试集评估
test_metric = trainer.evaluate(test_token)
print("=========测试集指标【复制到你的报告】==========")
print(test_metric)

# 输出模型文件夹 bert_live_speech_best，生成在项目根目录
trainer.save_model("./bert_live_speech_best")
tokenizer.save_pretrained("./bert_live_speech_best")
