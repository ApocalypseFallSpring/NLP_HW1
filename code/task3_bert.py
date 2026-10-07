"""
Phase 4 / Task 3: Fine-tune BERT for NYT text classification
============================================================
作业要求：
  - 预训练模型: google-bert/bert-base-uncased
  - max_length = 64
  - epochs = 3
  - 在 NYT Test Set 上报告 Accuracy / Macro-F1

Pipeline（和 BoW / Word2Vec 的对比）：
  BoW/W2V : 文本 → 手工/自监督特征 → 另训 LR
  BERT    : 文本 → Tokenizer → BERT(整模可更新) → 分类头 → 直接出类别
            也就是「表示 + 分类」一起 fine-tune。

国内下载模型建议设置环境变量（脚本里也会尝试设置）：
  HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import os
from pathlib import Path

# 尽量走 HuggingFace 镜像，避免官方源超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from metrics import report

# ---------------------------------------------------------------------------
# 超参（作业硬性要求 + 常用微调设置）
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "bert"
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # BERT 微调常用学习率；作业未指定，用社区默认值


class NytDataset(Dataset):
    """把 (input_ids, attention_mask, label) 包成 PyTorch Dataset。"""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_texts(tokenizer, texts: list[str]) -> dict:
    """
    BERT Tokenizer：
      - 切成 WordPiece subwords
      - 截断/填充到 max_length=64
      - 返回 input_ids / attention_mask
    """
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # 动态 padding 交给 data collator，更省算力
    )


def compute_metrics(eval_pred):
    """Trainer 在每个 epoch 结束时调用；我们同时看 Acc 和 Macro-F1。"""
    from metrics import accuracy, macro_f1

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy(labels, preds),
        "macro_f1": macro_f1(labels, preds),
    }


@torch.no_grad()
def predict_labels(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    out = trainer.predict(dataset)
    return np.argmax(out.predictions, axis=-1)


def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    print(f"[info] model = {MODEL_NAME}")
    print(f"[info] max_length = {MAX_LENGTH}, epochs = {NUM_EPOCHS}, lr = {LEARNING_RATE}")

    # ----- 读 Phase1 固定划分 -----
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # 标签 → 整数 id（BERT 分类头需要）
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str)).tolist()
    y_val = le.transform(val_df["label"].astype(str)).tolist()
    y_test = le.transform(test_df["label"].astype(str)).tolist()
    label_list = list(le.classes_)
    id2label = {i: lab for i, lab in enumerate(label_list)}
    label2id = {lab: i for i, lab in enumerate(label_list)}
    print(f"[info] labels = {label_list}")
    print(f"[info] train/val/test = {len(y_train)}/{len(y_val)}/{len(y_test)}")

    # ----- Tokenizer & 编码 -----
    print("[info] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenize_texts(tokenizer, train_df["text"].astype(str).tolist())
    val_enc = tokenize_texts(tokenizer, val_df["text"].astype(str).tolist())
    test_enc = tokenize_texts(tokenizer, test_df["text"].astype(str).tolist())

    train_ds = NytDataset(train_enc, y_train)
    val_ds = NytDataset(val_enc, y_val)
    test_ds = NytDataset(test_enc, y_test)

    # ----- 模型：预训练 BERT + 随机初始化的分类头 -----
    print("[info] 加载 BERT ForSequenceClassification ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # ----- 训练参数 -----
    # evaluation_strategy=epoch：每个 epoch 在 val 上看一眼，方便你观察有没有过拟合
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        report_to=[],  # 不上报 wandb 等
        fp16=torch.cuda.is_available(),  # GPU 上开半精度加速
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[info] 开始 fine-tune ...")
    train_result = trainer.train()
    print(f"[info] train done. metrics = {train_result.metrics}")

    # ----- 最终评估：train / val / test -----
    print("\n===== 最终指标 =====")
    results = {}
    for name, ds, y_true in [
        ("train", train_ds, y_train),
        ("val", val_ds, y_val),
        ("test", test_ds, y_test),
    ]:
        preds = predict_labels(trainer, ds)
        results[name] = report(y_true, preds, name=f"BERT/{name}")

    # 保存模型，方便以后只做 inference
    save_dir = OUT_DIR / "best_model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[info] 模型已保存 → {save_dir}")

    print("\n===== 汇总（作业重点：Test）=====")
    print(
        f"BERT-base-uncased | Test Acc={results['test']['accuracy']:.4f} "
        f"| Test Macro-F1={results['test']['macro_f1']:.4f}"
    )
    print(
        f"  val : Acc={results['val']['accuracy']:.4f}  "
        f"Macro-F1={results['val']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
