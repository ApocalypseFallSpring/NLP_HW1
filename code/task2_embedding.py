"""
Phase 3 / Task 2: Word Embedding + Logistic Regression
======================================================
三组实验（文档向量都是「有效词向量取平均」，再接同一个 LR）：

  1) Pre-trained GloVe (glove.6B.100d)
  2) Word2Vec 在 AG News 上自训练 (100 维)
  3) Word2Vec 在 NYT 训练集上自训练 (100 维)

文档向量公式：
  e(d) = (1/n) * sum_i e(w_i)
  其中只对「在 embedding 里找得到的词」求平均；一个有效词都没有 → 用零向量。

和 Task1 BoW 的直觉对比：
  BoW：每个词是词表里的一维（稀疏、高维）
  Embedding：每个词是稠密 100 维语义向量，文档再平均成 100 维
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LogisticRegression

# 复用 Task1 的分词（含 NLTK / 缓存逻辑）
from task1_bow import DATA_DIR, OUT_DIR, SEED, tokenize_corpus
from metrics import ce_loss, report

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = Path(__file__).resolve().parent / "embeddings"
GLOVE_TXT = EMB_DIR / "glove.6B.100d.txt"
GLOVE_GENSIM_GZ = EMB_DIR / "glove-wiki-gigaword-100.gz"  # 与 glove.6B.100d 同源
AG_PATH = ROOT / "ag.csv"

DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 5
W2V_SG = 0  # 0=CBOW（快）；1=Skip-gram。作业未指定，用 CBOW
LR_MAX_ITER = 2000


# ---------------------------------------------------------------------------
# Embedding 查表 & 文档平均
# ---------------------------------------------------------------------------
class EmbeddingTable:
    """统一接口：word -> 100 维向量。不在表里的词 = 无效词，跳过。"""

    def __init__(self, vectors: dict[str, np.ndarray], name: str):
        self.vectors = vectors
        self.name = name
        self.dim = DIM

    def __contains__(self, word: str) -> bool:
        return word in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def get(self, word: str) -> np.ndarray | None:
        return self.vectors.get(word)


def load_glove() -> EmbeddingTable:
    """
    加载 100 维预训练 GloVe（glove.6B / wiki-gigaword-100，同源）。
    优先读本地 txt；否则读 gensim 发布的 .gz（word2vec 格式）。
    """
    if GLOVE_TXT.exists():
        print(f"[info] 加载 GloVe txt: {GLOVE_TXT}")
        vectors: dict[str, np.ndarray] = {}
        with GLOVE_TXT.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != DIM + 1:
                    continue
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        print(f"[info] GloVe 词数 = {len(vectors)}")
        return EmbeddingTable(vectors, name="GloVe")

    if GLOVE_GENSIM_GZ.exists():
        print(f"[info] 加载 GloVe gensim gz: {GLOVE_GENSIM_GZ}")
        kv = KeyedVectors.load_word2vec_format(str(GLOVE_GENSIM_GZ), binary=False)
        vectors = {w: kv[w].astype(np.float32) for w in kv.index_to_key}
        print(f"[info] GloVe 词数 = {len(vectors)}")
        return EmbeddingTable(vectors, name="GloVe")

    raise FileNotFoundError(
        "找不到 GloVe 文件。请运行:\n"
        "  python download_glove.py\n"
        f"期望路径之一:\n  {GLOVE_TXT}\n  {GLOVE_GENSIM_GZ}"
    )

def train_word2vec(sentences: list[list[str]], name: str, save_path: Path) -> EmbeddingTable:
    """
    用 gensim 训练 Word2Vec，再转成 EmbeddingTable。

    sentences: 已经分好词的文档列表，例如 [["wall","st","bears"], ...]
    """
    if save_path.exists():
        print(f"[info] 加载已训练 Word2Vec: {save_path}")
        model = Word2Vec.load(str(save_path))
    else:
        print(f"[info] 训练 Word2Vec on {name}: docs={len(sentences)}, dim={DIM} ...")
        model = Word2Vec(
            sentences=sentences,
            vector_size=DIM,
            window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT,
            workers=4,
            sg=W2V_SG,
            seed=SEED,
            epochs=W2V_EPOCHS,
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"[info] 已保存 → {save_path}")

    # KeyedVectors: word -> vector
    kv = model.wv
    vectors = {word: kv[word].astype(np.float32) for word in kv.index_to_key}
    print(f"[info] Word2Vec({name}) 词数 = {len(vectors)}")
    return EmbeddingTable(vectors, name=f"W2V-{name}")


def docs_to_matrix(tokenized_docs: list[list[str]], table: EmbeddingTable) -> np.ndarray:
    """
    每篇文档 → 100 维平均向量。
    有效词：分词结果里、且能在 embedding 表中查到的词。
    """
    X = np.zeros((len(tokenized_docs), table.dim), dtype=np.float32)
    n_empty = 0
    coverages = []

    for i, toks in enumerate(tokenized_docs):
        vecs = [table.vectors[w] for w in toks if w in table.vectors]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
            coverages.append(len(vecs) / max(len(toks), 1))
        else:
            n_empty += 1
            coverages.append(0.0)

    print(
        f"[info] 文档向量化完成 | 空文档(全OOV)={n_empty}/{len(tokenized_docs)} "
        f"| 平均词覆盖率={np.mean(coverages):.3f}"
    )
    return X


# ---------------------------------------------------------------------------
# 训练 LR + 评估
# ---------------------------------------------------------------------------
def run_experiment(
    exp_name: str,
    table: EmbeddingTable,
    train_toks: list[list[str]],
    val_toks: list[list[str]],
    test_toks: list[list[str]],
    y_train: list[str],
    y_val: list[str],
    y_test: list[str],
) -> dict:
    print(f"\n===== {exp_name} =====")
    print(f"[info] embedding 词表大小 = {len(table)}")

    X_train = docs_to_matrix(train_toks, table)
    X_val = docs_to_matrix(val_toks, table)
    X_test = docs_to_matrix(test_toks, table)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    print("[info] 训练 Logistic Regression ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(X_train, y_train)

    results = {}
    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        m = report(y, pred, name=f"{exp_name}/{split}")
        m["log_loss"] = ce_loss(y, proba, labels=clf.classes_)
        print(f"[{exp_name}/{split}] log-loss = {m['log_loss']:.6f}")
        results[split] = m
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- NYT 划分（与 Phase1 固定一致）-----
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    y_train = train_df["label"].astype(str).tolist()
    y_val = val_df["label"].astype(str).tolist()
    y_test = test_df["label"].astype(str).tolist()

    train_toks = tokenize_corpus(train_df["text"].astype(str).tolist(), "train")
    val_toks = tokenize_corpus(val_df["text"].astype(str).tolist(), "val")
    test_toks = tokenize_corpus(test_df["text"].astype(str).tolist(), "test")

    all_results = {}

    # ----- 1) Pre-trained GloVe -----
    glove = load_glove()
    all_results["glove"] = run_experiment(
        "GloVe-6B-100d", glove,
        train_toks, val_toks, test_toks, y_train, y_val, y_test,
    )

    # ----- 2) Word2Vec on AG News -----
    # AG 只有 text，无监督语料，用来学词向量，再迁到 NYT 文档上
    ag_df = pd.read_csv(AG_PATH)
    ag_toks = tokenize_corpus(ag_df["text"].astype(str).tolist(), "ag")
    w2v_ag = train_word2vec(ag_toks, name="AG", save_path=OUT_DIR / "w2v_ag.model")
    all_results["w2v_ag"] = run_experiment(
        "Word2Vec-AG", w2v_ag,
        train_toks, val_toks, test_toks, y_train, y_val, y_test,
    )

    # ----- 3) Word2Vec on NYT train -----
    # 只用训练集文本训 embedding，避免用 test 文本更新词向量（更干净）
    w2v_nyt = train_word2vec(train_toks, name="NYT-train", save_path=OUT_DIR / "w2v_nyt.model")
    all_results["w2v_nyt"] = run_experiment(
        "Word2Vec-NYT", w2v_nyt,
        train_toks, val_toks, test_toks, y_train, y_val, y_test,
    )

    # ----- 汇总 -----
    print("\n===== 汇总（重点看 Test）=====")
    for key, title in [
        ("glove", "1. Pre-trained GloVe"),
        ("w2v_ag", "2. Word2Vec on AG"),
        ("w2v_nyt", "3. Word2Vec on NYT"),
    ]:
        r = all_results[key]
        print(
            f"{title}\n"
            f"  val : Acc={r['val']['accuracy']:.4f}  Macro-F1={r['val']['macro_f1']:.4f}\n"
            f"  test: Acc={r['test']['accuracy']:.4f}  Macro-F1={r['test']['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
