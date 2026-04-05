from ultralytics import YOLO
import multiprocessing


def train_emotion_model():
    print("🔥 启动 YOLO 表情分类模型训练引擎...")

    # 1. 加载预训练模型作为起点 (Transfer Learning)
    local_weights_path = './weights/yolo26n-cls.pt'

    print(f"尝试加载本地预训练权重: {local_weights_path}")
    model = YOLO(local_weights_path)

    # 2. 核心训练配置
    results = model.train(
        data='datasets/rafdb_yolo',  # 我们刚刚精心整理的数据集根目录
        task='classify',
        epochs=100,  # 训练轮数：先跑 50 轮看看收敛趋势
        imgsz=128,  # 图像尺寸：分类任务的标准输入大小
        batch=64,  # 批次大小：如果你的电脑显存/内存够大，可以开到 64 或 128
        cache="ram", #训练之前先将图片从硬盘中提前读取到内存中
        workers=8,  # 数据加载线程数，优化数据读取速度
        device='',  # 留空代表自动选择 (优先 CUDA，其次 MPS，最后 CPU)
        # project='runs/classify',  # 训练日志和权重的保存主目录
        name='fer_baseline',  # 本次实验的名称
        pretrained=True,  # 开启预训练权重，加速收敛并提高泛化能力

        # --- 数据增强策略 (对抗网课复杂环境的法宝) ---
        fliplr=0.5,  # 50% 概率水平翻转 (增强面部左右对称泛化)
        hsv_h=0.015,  # 色调微调 (对抗不同房间的光线变化)
        hsv_s=0.7,  # 饱和度微调
        hsv_v=0.4  # 亮度微调
    )

    print("🎉 训练任务执行完毕！")
    print("最佳权重文件 (best.pt) 已保存在: runs/classify/fer_baseline/weights/ 目录下。")


if __name__ == '__main__':
    # 在 Windows/macOS 下，多进程训练必须放在 __main__ 保护块中
    train_emotion_model()