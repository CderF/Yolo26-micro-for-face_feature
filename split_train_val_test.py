"""
把 datasets/rafdb_yolo 从 (train / val) 两份划分，重整成真正独立的三份：

    train/  <- 从原 train 里分层抽样留下的大头，训练用
    val/    <- 从原 train 里分层抽样切出来的一小部分 (默认15%)，训练期间早停/挑 best.pt 用
    test/   <- 原来的 val (也就是 RAF-DB 官方 test 集)，原封不动地封存起来，
               只在最后要写进论文的时候跑一次 evaluate_model.py --split test，
               中途不看它的结果做任何决策

关键点 —— 按"来源分组"切分，而不是按文件切分：
    augment_rare_classes.py 生成的增强图片文件名是 {原图stem}_aug{N}.jpg，
    和它的原图是同一张脸的不同增强版本，长得很像。如果切分时把某张原图分进
    train、它的增强版本分进 val (或反过来)，val 就会混入训练时"看过的"图片的
    近似变体，早停判断会被这种数据泄漏污染，选出来的 best.pt 不可信。
    所以这里先把同源的文件 (原图 + 它所有的 _aug* 版本) 分成一组，按组做
    分层抽样，保证一组图片要么整组进 train，要么整组进 val，不会被拆开。

这是一次性的重排操作 (原地 move，不是 copy)，运行前建议先确认已经有一份
数据集备份 (比如用 cp -R 拷到 git 仓库之外的地方)，脚本也会做一个安全检查：
如果 test/ 已经存在就直接拒绝跑，避免把已经切分过的状态又切一遍。

用法:
    python split_train_val_test.py                  # 默认 val 占15%
    python split_train_val_test.py --val-ratio 0.10  # 想要10%就自己调
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
AUG_SUFFIX_RE = re.compile(r"^(?P<base>.+)_aug\d+$")


def source_key(path: Path) -> str:
    """同源分组的 key：原图和它所有 _aug{N} 增强版本会得到同一个 key。"""
    m = AUG_SUFFIX_RE.match(path.stem)
    return m.group("base") if m else path.stem


def split_class(class_dir: Path, val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    files = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        groups[source_key(f)].append(f)

    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    n_val_groups = max(1, round(len(group_keys) * val_ratio)) if group_keys else 0
    val_keys = set(group_keys[:n_val_groups])

    train_files, val_files = [], []
    for key, flist in groups.items():
        (val_files if key in val_keys else train_files).extend(flist)
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="把 train/val 两份划分重整为 train/val/test 三份独立划分")
    parser.add_argument("--data", type=str, default="datasets/rafdb_yolo", help="数据集根目录")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="从原 train 里切多少比例出来当新 val (按来源分组计算)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data)
    old_train_dir = root / "train"
    old_val_dir = root / "val"  # 目前是 RAF-DB 官方 test 集
    new_test_dir = root / "test"

    if new_test_dir.exists():
        raise SystemExit(
            f"❌ {new_test_dir} 已经存在，说明可能已经切分过了，为避免重复处理已中止。"
            f"\n   如果想用不同的 --val-ratio 重新切，请先从备份恢复 {root} 再重新运行。"
        )
    if not old_train_dir.is_dir() or not old_val_dir.is_dir():
        raise SystemExit(f"❌ 没找到 {old_train_dir} 或 {old_val_dir}，请检查 --data 路径")

    class_names = sorted(p.name for p in old_train_dir.iterdir() if p.is_dir())
    print(f"检测到 {len(class_names)} 个类别: {class_names}")

    # 1. 原 val (RAF-DB 官方 test 集) 原地封存成 test，不做任何改动
    print(f"\n📦 封存原 val ({old_val_dir}) -> test ({new_test_dir}) ...")
    shutil.move(str(old_val_dir), str(new_test_dir))

    # 2. 原 train 按来源分组分层抽样，切出新的 val
    new_val_dir = root / "val"
    new_val_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"\n✂️  按 val_ratio={args.val_ratio} 对 train 做分层抽样 (按同源分组，增强图片不会和原图被拆开)...")
    summary = []
    for cls in class_names:
        class_dir = old_train_dir / cls
        train_files, val_files = split_class(class_dir, args.val_ratio, rng)

        dst_dir = new_val_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in val_files:
            shutil.move(str(f), str(dst_dir / f.name))

        summary.append((cls, len(train_files), len(val_files)))
        print(f"  {cls:<10} train={len(train_files):>5}  val={len(val_files):>5}")

    # 3. 清掉过期缓存，下次训练会自动重新扫描
    for cache in (root / "train.cache", root / "val.cache", root / "test.cache"):
        if cache.exists():
            cache.unlink()
            print(f"🗑️  删除过期缓存: {cache}")

    total_train = sum(s[1] for s in summary)
    total_val = sum(s[2] for s in summary)
    total_test = len(list(new_test_dir.rglob("*.jpg")))
    print(f"\n🎉 三份划分完成:")
    print(f"  train: {total_train} 张  (训练用)")
    print(f"  val:   {total_val} 张  (训练期间早停/挑 best.pt 用，来自原 train 切分)")
    print(f"  test:  {total_test} 张  (RAF-DB 官方 test 集，原封不动封存，只在最终报告论文数字时用一次)")


if __name__ == "__main__":
    main()
