"""
在 train_classifier.py 的基础上，加入两种独立的类别不均衡应对手段：

1. USE_WEIGHTED_LOSS: 类别加权 loss —— 稀有类 (Fear/Disgust) 算错时的梯度权重更高
2. USE_OVERSAMPLING:  过采样 —— 训练时稀有类图片被抽到的概率更高，让模型多看几遍

两者默认都关闭 (对齐 train_classifier.py 的基线行为)，按需要打开一个来试。
不建议同时打开两个 —— 两种手段都是在"放大稀有类的存在感"，叠加使用容易过度
矫正，把 Happy 这种多数类的效果拉下来，可以自己对比 evaluate_model.py 的 F1
数据看要不要都开。

用法:
    python train_classifier_balanced.py
    (在文件顶部的 USE_WEIGHTED_LOSS / USE_OVERSAMPLING / WEIGHT_MODE 里切换)
"""
import multiprocessing
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER

# ============ 在这里切换要试的方案 ============
USE_WEIGHTED_LOSS = True  # 类别加权 loss
USE_OVERSAMPLING = False  # 过采样 (不建议和上面同时开)
WEIGHT_MODE = "sqrt_inv"  # "sqrt_inv" (温和) 或 "inv" (力度更强，稀有类样本量差距很大时容易压过头)
# ===============================================


def compute_class_weights(counts: dict[int, int], mode: str = "sqrt_inv") -> torch.Tensor:
    """按类别样本数算加权系数，均值归一化为1，不整体改变loss的量级。"""
    ordered = torch.tensor([counts[i] for i in sorted(counts)], dtype=torch.float)
    if mode == "inv":  # 等价于 sklearn class_weight='balanced'，纠偏力度最强
        w = ordered.sum() / (len(ordered) * ordered)
    elif mode == "sqrt_inv":  # 取根号，纠偏更温和，避免多数类被压制太狠
        w = torch.sqrt(ordered.sum() / (len(ordered) * ordered))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return w / w.mean()


class WeightedClassificationLoss:
    """和 ultralytics.utils.loss.v8ClassificationLoss 等价，只是多传一个 per-class weight。"""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        # AMP 混合精度下 preds 可能是 float16，weight 必须转成同精度，否则 cross_entropy 报
        # "expected scalar type Half but found Float"
        weight = self.weight.to(device=preds.device, dtype=preds.dtype)
        loss = F.cross_entropy(preds, batch["cls"], weight=weight, reduction="mean")
        return loss, loss.detach()


class BalancedClassificationTrainer(ClassificationTrainer):
    """在标准 ClassificationTrainer 上挂载加权loss / 过采样，均为可选。"""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, weights, verbose)
        if USE_WEIGHTED_LOSS:
            counts = self._train_class_counts()
            class_weight = compute_class_weights(counts, WEIGHT_MODE)
            model.criterion = WeightedClassificationLoss(class_weight)
            names = self.data["names"]
            LOGGER.info(
                "启用类别加权 loss (" + WEIGHT_MODE + "): "
                + ", ".join(f"{names[i]}={class_weight[i]:.2f}" for i in sorted(names))
            )
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train" or not USE_OVERSAMPLING:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        dataset = self.build_dataset(dataset_path, mode)
        labels = [s[1] for s in dataset.samples]
        counts = Counter(labels)
        sample_weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        batch_size = min(batch_size, len(dataset))
        nw = min(os.cpu_count() or 1, self.args.workers)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205)
        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # 用 sampler 时必须是 False
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        names = self.data["names"]
        LOGGER.info(f"启用过采样，各类别抽样概率已按 1/样本数 重新加权: {sorted(counts.items())} -> {names}")
        return loader

    def _train_class_counts(self) -> dict[int, int]:
        train_dir = Path(self.data["train"])
        names = self.data["names"]
        return {i: len(list((train_dir / names[i]).glob("*"))) for i in names}


def train_emotion_model():
    print("🔥 启动 YOLO 表情分类模型训练引擎 (类别不均衡应对版) ...")
    print(f"USE_WEIGHTED_LOSS={USE_WEIGHTED_LOSS}  USE_OVERSAMPLING={USE_OVERSAMPLING}  WEIGHT_MODE={WEIGHT_MODE}")

    custom_yaml = "ultralytics/cfg/models/26/yolo26-micro.yaml"
    local_weights_path = "./weights/yolo26n-cls.pt"

    model = YOLO(custom_yaml).load(local_weights_path)

    model.train(
        trainer=BalancedClassificationTrainer,
        data="datasets/rafdb_yolo",
        task="classify",
        epochs=150,
        patience=30,
        imgsz=128,
        batch=64,
        cache="ram",
        workers=8,
        device="",
        name="fer_micro_cls_balanced",
        pretrained=False,
        lr0=0.001,
        warmup_epochs=5,
        cos_lr=True,
        dropout=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("🎉 训练任务执行完毕！")
    print("最佳权重文件 (best.pt) 已保存在: runs/classify/fer_micro_cls_balanced/weights/ 目录下。")
    print("训练完用 evaluate_model.py 对比一下 Fear/Disgust 的 F1 有没有真的改善。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_emotion_model()
