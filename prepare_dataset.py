import os
import shutil
import pandas as pd
from pathlib import Path


def prepare_yolo_classification_dataset(raw_dir='rafdb_raw', target_dir='datasets/rafdb_yolo'):
    """
    将包含 CSV 标签的 RAF-DB 原始数据集转换为 YOLO-cls 所需的 ImageNet 文件夹层级结构。
    完美适配 images/train/数字/图片.jpg 的终极嵌套结构。
    """
    print("🚀 开始构建 YOLO26-cls 训练数据集结构...")

    # RAF-DB 的标准标签映射
    emotion_map = {
        1: 'Surprise',
        2: 'Fear',
        3: 'Disgust',
        4: 'Happy',
        5: 'Sad',
        6: 'Anger',
        7: 'Neutral'
    }

    # 路径定义
    raw_images_dir = Path(raw_dir) / 'images'
    train_csv = Path(raw_dir) / 'train_labels.csv'
    test_csv = Path(raw_dir) / 'test_labels.csv'

    target_base = Path(target_dir)

    if not raw_images_dir.exists() or not train_csv.exists() or not test_csv.exists():
        print(f"❌ 错误：请确保 {raw_dir} 目录下包含 images 文件夹以及两个 csv 文件！")
        return

    def process_split(csv_file, target_split_name, source_subfolder):
        df = pd.read_csv(csv_file)
        total_imgs = len(df)
        success_count = 0
        missing_count = 0

        print(f"\n📂 正在处理 {target_split_name} 集 (共 {total_imgs} 张)...")

        for index, row in df.iterrows():
            img_name = str(row['image']).strip()
            label_idx = int(row['label'])

            # 获取对应的英文表情名称
            class_name = emotion_map.get(label_idx, 'Unknown')

            # 创建目标目录，例如：datasets/rafdb_yolo/train/Happy/
            class_dir = target_base / target_split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # ====== 终极修改：将 label_idx (1-7的数字) 转换为字符串，加入路径拼接中 ======
            # 这样就能精准定位到 rafdb_raw/images/train/5/train_00001_aligned.jpg
            src_path = raw_images_dir / source_subfolder / str(label_idx) / img_name
            dst_path = class_dir / img_name
            # =======================================================================

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                success_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"❌ 找不到文件，我试图寻找的路径是: {src_path.absolute()}")

        print(f"✅ {target_split_name} 集处理完成: 成功移动 {success_count} 张，缺失 {missing_count} 张。")

    # 执行划分
    process_split(train_csv, 'train', 'train')
    process_split(test_csv, 'val', 'test')

    print(f"\n🎉 数据集格式化大功告成！目标路径: {target_base.absolute()}")


if __name__ == '__main__':
    prepare_yolo_classification_dataset()