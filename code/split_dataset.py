from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "nyt.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"

SEED = 42  # 固定随机种子，保证以后每次划分结果完全一样


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"[info] 原始数据: {df.shape[0]} 行, 列 = {list(df.columns)}")
    print("[info] 全量 label 分布:")
    print(df["label"].value_counts().sort_index())
    print()

    # sample(frac=1) 表示打乱全部行
    # random_state=SEED 保证可复现
    # reset_index(drop=True) 丢掉旧索引，重新从 0 编号
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n = len(df)
    n_train = n * 8 // 10
    n_val = n * 1 // 10

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.csv"
    val_path = OUT_DIR / "val.csv"
    test_path = OUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("[info] 划分结果:")
    for name, subset, path in [
        ("train", train_df, train_path),
        ("val", val_df, val_path),
        ("test", test_df, test_path),
    ]:
        print(f"  {name}: {len(subset)} 条 → {path}")
        print(f"    label 分布:\n{subset['label'].value_counts().sort_index().to_string()}")
        print()

    print("[done] 后续任务请统一读取 code/data/ 下的这三个文件。")


if __name__ == "__main__":
    main()
