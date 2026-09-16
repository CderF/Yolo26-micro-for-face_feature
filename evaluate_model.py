"""
加载训练好的分类权重，在指定数据集划分上跑推理，输出每类 precision/recall/F1、
macro-F1、weighted-F1，并把结果追加写入 CSV，方便横向对比不同训练权重的效果。

用法示例:
    python evaluate_model.py --weights runs/classify/fer_micro_cls/weights/best.pt
    python evaluate_model.py --weights /path/to/other_run/best.pt --split val
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def compute_prf1(matrix: np.ndarray) -> dict:
    """从 (nc, nc) 混淆矩阵 (行=预测类别, 列=真实类别) 计算每类 precision/recall/F1。"""
    tp = np.diag(matrix)
    support = matrix.sum(axis=0)  # 每类真实样本数
    pred_count = matrix.sum(axis=1)  # 每类被预测的次数

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp, dtype=float), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)

    accuracy = tp.sum() / matrix.sum() if matrix.sum() > 0 else 0.0
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def evaluate(weights: str, data: str, split: str, imgsz: int, batch: int, device: str) -> tuple[dict, dict]:
    model = YOLO(weights)
    val_results = model.val(
        data=data, split=split, imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False
    )

    matrix = val_results.confusion_matrix.matrix  # (nc, nc)，行=预测 列=真实
    names = model.names  # {idx: class_name}
    stats = compute_prf1(matrix)

    print(f"\n{'=' * 60}")
    print(f"权重: {weights}")
    print(f"数据: {data}/{split}  imgsz={imgsz}  样本数: {int(matrix.sum())}")
    print(f"{'=' * 60}")
    print(f"{'类别':<12}{'support':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    for i in sorted(names.keys()):
        print(
            f"{names[i]:<12}{int(stats['support'][i]):>8}{stats['precision'][i]:>11.4f}"
            f"{stats['recall'][i]:>9.4f}{stats['f1'][i]:>8.4f}"
        )
    print(f"{'-' * 60}")
    print(f"accuracy (top1): {stats['accuracy']:.4f}")
    print(f"macro F1:        {stats['macro_f1']:.4f}")
    print(f"weighted F1:     {stats['weighted_f1']:.4f}")

    return stats, names


def log_result(log_path: str, weights: str, data: str, split: str, stats: dict, names: dict) -> None:
    log_path = Path(log_path)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            header = ["timestamp", "weights", "data", "split", "n_images", "accuracy", "macro_f1", "weighted_f1"]
            header += [f"f1_{names[i]}" for i in sorted(names.keys())]
            writer.writerow(header)
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            weights,
            data,
            split,
            int(stats["support"].sum()),
            f"{stats['accuracy']:.4f}",
            f"{stats['macro_f1']:.4f}",
            f"{stats['weighted_f1']:.4f}",
        ]
        row += [f"{stats['f1'][i]:.4f}" for i in sorted(names.keys())]
        writer.writerow(row)
    print(f"\n结果已追加写入: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估分类模型权重，计算F1等指标，便于横向对比不同训练结果")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径，例如 runs/classify/xxx/weights/best.pt")
    parser.add_argument("--data", type=str, default="datasets/rafdb_yolo", help="数据集根目录")
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "train", "test"],
        help="评估用的子集：val=训练期间用来挑best.pt的验证集；test=封存起来的RAF-DB官方test集，只在最终报告论文数字时用",
    )
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="", help="留空自动选择 (CUDA > MPS > CPU)")
    parser.add_argument("--log", type=str, default="eval_log.csv", help="结果追加写入的对比记录文件")
    args = parser.parse_args()

    result_stats, class_names = evaluate(args.weights, args.data, args.split, args.imgsz, args.batch, args.device)
    log_result(args.log, args.weights, args.data, args.split, result_stats, class_names)
