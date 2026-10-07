from __future__ import annotations

import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from metrics import ce_loss, report

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
# 逐轮记录 loss：每轮只走 1 次迭代（warm_start），便于画出训练曲线
MAX_EPOCHS = 120          # 用于画出 loss 曲线的逐轮迭代上限
LOSS_TOL = 1e-5           # 相邻两轮 train loss 变化小于此值就结束逐轮记录
FINAL_MAX_ITER = 2000     # 曲线记完后，再充分收敛一次，作为最终模型


# ---------------------------------------------------------------------------
# 1) 分词
# ---------------------------------------------------------------------------
def _nltk_ok() -> bool:
    try:
        from nltk import word_tokenize

        word_tokenize("test")
        return True
    except Exception:
        return False


def _download_punkt_via_mirror() -> bool:
    import io
    import zipfile
    from urllib.request import urlopen

    import nltk

    roaming = Path.home() / "AppData" / "Roaming" / "nltk_data" / "tokenizers"
    tok_dir = roaming if roaming.parent.parent.exists() else Path(nltk.data.path[0]) / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)

    mirrors = [
        "https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers",
        "https://ghfast.top/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers",
    ]
    for pkg in ("punkt", "punkt_tab"):
        dest_dir = tok_dir / pkg
        if dest_dir.exists() and any(dest_dir.rglob("*")):
            continue
        ok = False
        for base in mirrors:
            url = f"{base}/{pkg}.zip"
            try:
                print(f"[info] 镜像下载 {pkg}: {url}")
                with urlopen(url, timeout=120) as resp:
                    raw = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    zf.extractall(tok_dir)
                ok = True
                print(f"[info] {pkg} 安装完成 → {dest_dir}")
                break
            except Exception as e:
                print(f"[warn] 镜像失败 ({base}): {e}")
        if not ok:
            return False
    return _nltk_ok()


def _ensure_nltk_punkt() -> bool:
    if _nltk_ok():
        return True
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        if _nltk_ok():
            return True
    except Exception:
        pass
    print("[info] 官方下载失败，尝试镜像 ...")
    return _download_punkt_via_mirror()


_USE_NLTK = _ensure_nltk_punkt()


def tokenize(text: str) -> list[str]:
    """文档 → token 列表。优先 nltk.word_tokenize；失败则正则兜底。"""
    text = str(text).lower()
    if _USE_NLTK:
        from nltk import word_tokenize

        return word_tokenize(text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def tokenize_corpus(texts: list[str], name: str) -> list[list[str]]:
    """整份语料分词一次；结果缓存到 outputs/，重跑时可跳过耗时分词。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / f"tokens_{name}.pkl"
    if cache_path.exists():
        print(f"[info] 加载分词缓存 {cache_path.name} ...")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print(f"[info] 分词 {name} ({len(texts)} 篇) ...")
    toks = [tokenize(t) for t in texts]
    with cache_path.open("wb") as f:
        pickle.dump(toks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[info] 已缓存分词 → {cache_path}")
    return toks


# ---------------------------------------------------------------------------
# 2) 词表（只看训练集的 token）
# ---------------------------------------------------------------------------
def build_vocabulary(tokenized_docs: list[list[str]]) -> dict[str, int]:
    vocab_set: set[str] = set()
    for toks in tokenized_docs:
        vocab_set.update(toks)
    return {word: i for i, word in enumerate(sorted(vocab_set))}


# ---------------------------------------------------------------------------
# 3) 文档 → 向量（两种 BoW）
# ---------------------------------------------------------------------------
def vectorize(
    tokenized_docs: list[list[str]],
    word2idx: dict[str, int],
    mode: str,
) -> csr_matrix:
    """
    mode="binary": 出现→1，否则→0
    mode="count" : 写入出现次数（Word Frequency）
    返回稀疏矩阵，形状 = (文档数, |V|)
    """
    assert mode in {"binary", "count"}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for row_id, tokens in enumerate(tokenized_docs):
        counts = Counter(tok for tok in tokens if tok in word2idx)
        for word, cnt in counts.items():
            rows.append(row_id)
            cols.append(word2idx[word])
            data.append(1.0 if mode == "binary" else float(cnt))

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokenized_docs), len(word2idx)),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 4) 训练：记录每一轮 train / val log-loss
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    用 warm_start + max_iter=1，把“每一轮优化”拆开，才能看到 loss 曲线。

    为什么默认一次 fit 看不到曲线？
      sklearn 的 LogisticRegression 默认一口气优化到底，不返回中间 loss。
      我们主动每轮只迭代 1 步，就能记录 train/val 的交叉熵。
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1,       # 每轮只走 1 次迭代
        warm_start=True,  # 下一轮接着上一轮的参数继续
        random_state=SEED,
    )

    history: list[dict[str, float]] = []
    labels = sorted(set(y_train) | set(y_val))

    print(f"[info] 开始训练（最多 {MAX_EPOCHS} 轮，记录 train/val log-loss）...")
    with warnings.catch_warnings():
        # 每轮 max_iter=1 会提示“还没收敛”，这是预期现象
        warnings.filterwarnings("ignore", category=UserWarning)
        for epoch in range(1, MAX_EPOCHS + 1):
            clf.fit(X_train, y_train)

            train_loss = ce_loss(y_train, clf.predict_proba(X_train), labels=labels)
            val_loss = ce_loss(y_val, clf.predict_proba(X_val), labels=labels)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
                print(f"  epoch {epoch:3d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            if epoch >= 2 and abs(history[-2]["train_loss"] - train_loss) < LOSS_TOL:
                print(f"[info] 逐轮记录早停于 epoch={epoch}（train_loss 变化 < {LOSS_TOL}）")
                break

    # 曲线只是为了“看见 loss 怎么降”；最终模型再充分优化一轮，避免 count 特征未收敛
    print(f"[info] 最终收敛训练 max_iter={FINAL_MAX_ITER} ...")
    clf.set_params(max_iter=FINAL_MAX_ITER)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(X_train, y_train)
    final_train = ce_loss(y_train, clf.predict_proba(X_train), labels=labels)
    final_val = ce_loss(y_val, clf.predict_proba(X_val), labels=labels)
    history.append({
        "epoch": history[-1]["epoch"] + 1 if history else 1,
        "train_loss": final_train,
        "val_loss": final_val,
    })
    print(f"  final: train_loss={final_train:.6f}  val_loss={final_val:.6f}")

    return clf, history


def eval_split(clf, X, y, split_name: str) -> dict[str, float]:
    """在某个划分上报告 Acc / Macro-F1 / log-loss。"""
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    metrics = report(y, pred, name=split_name)
    loss = ce_loss(y, proba, labels=clf.classes_)
    print(f"[{split_name}] log-loss = {loss:.6f}")
    metrics["log_loss"] = loss
    return metrics


def save_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    print(f"[info] loss 曲线已保存: {path}")


# ---------------------------------------------------------------------------
# 5) 一种 BoW 方法：向量化 → 训练 → train/val/test 全报
# ---------------------------------------------------------------------------
def run_one_method(
    name: str,
    mode: str,
    word2idx: dict[str, int],
    train_toks: list[list[str]],
    val_toks: list[list[str]],
    test_toks: list[list[str]],
    y_train: list[str],
    y_val: list[str],
    y_test: list[str],
) -> dict:
    print(f"\n===== {name} (mode={mode}) =====")
    X_train = vectorize(train_toks, word2idx, mode=mode)
    X_val = vectorize(val_toks, word2idx, mode=mode)
    X_test = vectorize(test_toks, word2idx, mode=mode)
    print(f"[info] X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    clf, history = train_logistic_regression(X_train, y_train, X_val, y_val)
    save_loss_curve(history, OUT_DIR / f"task1_{mode}_loss_curve.csv")

    print(f"\n--- {name} 最终指标 ---")
    results = {
        "history": history,
        "train": eval_split(clf, X_train, y_train, f"{name}/train"),
        "val": eval_split(clf, X_val, y_val, f"{name}/val"),
        "test": eval_split(clf, X_test, y_test, f"{name}/test"),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "epochs": len(history),
    }
    return results


def main() -> None:
    print(f"[info] 分词后端: {'nltk.word_tokenize' if _USE_NLTK else 'regex-fallback'}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    y_train = train_df["label"].astype(str).tolist()
    y_val = val_df["label"].astype(str).tolist()
    y_test = test_df["label"].astype(str).tolist()

    print(f"[info] train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 分词只做一遍，后面两种 BoW 共用
    train_toks = tokenize_corpus(train_df["text"].astype(str).tolist(), "train")
    val_toks = tokenize_corpus(val_df["text"].astype(str).tolist(), "val")
    test_toks = tokenize_corpus(test_df["text"].astype(str).tolist(), "test")

    print("[info] 构建词表（仅训练集）...")
    word2idx = build_vocabulary(train_toks)
    print(f"[info] |V| = {len(word2idx)}")

    results = {}
    results["binary"] = run_one_method(
        "Binary BoW", "binary", word2idx,
        train_toks, val_toks, test_toks, y_train, y_val, y_test,
    )
    results["count"] = run_one_method(
        "Word Frequency", "count", word2idx,
        train_toks, val_toks, test_toks, y_train, y_val, y_test,
    )

    print("\n===== 汇总 =====")
    for key, title in [("binary", "Binary BoW"), ("count", "Word Frequency")]:
        r = results[key]
        print(f"\n[{title}] epochs={r['epochs']}")
        print(f"  final train_loss={r['final_train_loss']:.6f}  val_loss={r['final_val_loss']:.6f}")
        for split in ("train", "val", "test"):
            m = r[split]
            print(
                f"  {split:5s}: Acc={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}  "
                f"log-loss={m['log_loss']:.6f}"
            )


if __name__ == "__main__":
    main()
