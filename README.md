# NLP 2026 Fall · 作业一：文本分类

NYT 新闻文本分类实验：Bag of Words、Word Embedding（GloVe / Word2Vec）、BERT Fine-tuning。  
评价指标：Accuracy、Macro-F1（在 NYT Test Set 上报告）。

## 数据

将课程提供的 `nyt.csv`、`ag.csv` 放在本目录（与 `code/` 同级）。

## 环境

Python 3.10+（本机实验为 Anaconda 3.11）。安装依赖：

```bash
pip install -r requirements.txt
```

- BERT 建议安装 CUDA 版 PyTorch，并设置 HuggingFace 镜像，例如：`HF_ENDPOINT=https://hf-mirror.com`
- NLTK 分词需要 `punkt` / `punkt_tab`；国内若官方下载失败，脚本会尝试镜像

## 运行

在 `code/` 目录下依次执行：

```bash
python split_dataset.py      # 划分 NYT：train/val/test = 8:1:1，seed=42
python task1_bow.py          # Task 1：Binary BoW / Word Frequency + LR
python download_glove.py     # 下载 100 维 GloVe（若尚未下载）
python task2_embedding.py    # Task 2：GloVe / Word2Vec-AG / Word2Vec-NYT + LR
python task3_bert.py         # Task 3：bert-base-uncased，max_length=64，3 epochs
```

划分结果写入 `code/data/`；GloVe 写入 `code/embeddings/`；Word2Vec 与 BERT 权重写入 `code/outputs/`（均不纳入仓库，可复现生成）。

## 代码说明

| 文件 | 作用 |
|------|------|
| `code/split_dataset.py` | 固定种子打乱并划分数据集 |
| `code/metrics.py` | Accuracy / Macro-F1 / log-loss |
| `code/task1_bow.py` | 词袋 + Logistic Regression |
| `code/task2_embedding.py` | 词向量平均 + Logistic Regression |
| `code/download_glove.py` | 下载预训练 GloVe |
| `code/task3_bert.py` | BERT 微调 |

词表与 NYT 上的 Word2Vec 均只使用训练集，避免测试集泄漏。
