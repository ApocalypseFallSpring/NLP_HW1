from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, log_loss


def accuracy(y_true, y_pred) -> float:
    """分类正确数 / 总数。"""
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """各类 F1 的算术平均（类别等权）。"""
    return float(f1_score(y_true, y_pred, average="macro"))


def ce_loss(y_true, y_proba, labels=None) -> float:
    """交叉熵 / log loss（LR 优化的就是这个）。"""
    return float(log_loss(y_true, y_proba, labels=labels))


def report(y_true, y_pred, name: str = "") -> dict[str, float]:
    """一次性打印并返回 Accuracy / Macro-F1。"""
    acc = accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    prefix = f"[{name}] " if name else ""
    print(f"{prefix}Accuracy = {acc:.4f}  |  Macro-F1 = {mf1:.4f}")
    return {"accuracy": acc, "macro_f1": mf1}
