"""
离线扩充稀有类别 (默认 Fear / Disgust) 的训练图片数量。

对每张源图随机套用一组增强 (旋转/色彩抖动/水平翻转/轻微仿射)，生成新文件写回
同一个类别目录下，文件名带 _aug{i} 后缀，不会覆盖或删除原图。只处理 train
目录，不碰 val —— 验证集必须保持原始分布，不然 evaluate_model.py 算出来的
指标就失真了。

跑完这个脚本之后，正常用 train_classifier.py / train_classifier_balanced.py
训练就行，不需要额外配置，多出来的图片会被当成普通训练数据的一部分。

用法:
    python augment_rare_classes.py --classes Fear Disgust --target-count 1500
    python augment_rare_classes.py --classes Fear --multiplier 4   # 按倍数而不是绝对数量
"""
import argparse
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms

AUGMENT_PIPELINE = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=6),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    ]
)


def augment_class(class_dir: Path, target_count: int) -> int:
    src_images = sorted([p for p in class_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not src_images:
        print(f"⚠️  {class_dir} 下没有图片，跳过")
        return 0

    current = len(src_images)
    need = target_count - current
    if need <= 0:
        print(f"✅ {class_dir.name}: 已有 {current} 张 >= 目标 {target_count}，跳过")
        return 0

    print(f"🎨 {class_dir.name}: 当前 {current} 张，目标 {target_count} 张，需要新增 {need} 张")
    generated = 0
    src_idx = 0
    while generated < need:
        src_path = src_images[src_idx % len(src_images)]
        src_idx += 1
        img = Image.open(src_path).convert("RGB")
        aug_img = AUGMENT_PIPELINE(img)
        out_path = src_path.with_name(f"{src_path.stem}_aug{generated}{src_path.suffix}")
        aug_img.save(out_path, quality=95)
        generated += 1
    print(f"   -> 生成完毕，{class_dir.name} 现有 {current + generated} 张")
    return generated


def main():
    parser = argparse.ArgumentParser(description="离线扩充稀有类别的训练图片数量")
    parser.add_argument("--data", type=str, default="datasets/rafdb_yolo", help="数据集根目录")
    parser.add_argument("--split", type=str, default="train", help="只对这个子集做增强 (不要用在 val 上)")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="要扩充的类别名，如 Fear Disgust")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-count", type=int, help="每个类别扩充到的目标图片数量")
    group.add_argument("--multiplier", type=float, help="按当前数量的倍数扩充，如 3 表示扩到3倍")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    split_dir = Path(args.data) / args.split
    total_generated = 0
    for cls_name in args.classes:
        class_dir = split_dir / cls_name
        if not class_dir.is_dir():
            print(f"❌ 找不到目录: {class_dir}")
            continue
        current = len([p for p in class_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        target = args.target_count if args.target_count is not None else round(current * args.multiplier)
        total_generated += augment_class(class_dir, target)

    print(f"\n🎉 全部完成，共新增 {total_generated} 张增强图片。")
    print("提醒: 如果之前跑过训练生成过 train.cache，需要删掉它 (或改文件名) 让 Ultralytics 重新扫描目录。")


if __name__ == "__main__":
    main()
