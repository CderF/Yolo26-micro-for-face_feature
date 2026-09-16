from ultralytics import YOLO
import multiprocessing


def train_emotion_model():
    print("yolo表情分类模型开始训练...")

    # 1. 使用自定义 yolo26-micro 结构 (P2+C2PSA, 截断P5, Classify头)
    #    并尝试挂载官方 ImageNet 预训练权重做部分迁移。
    #    注意：由于 C2PSA 被提前插入到了 P2 之后，骨干层的序号/通道数和官方结构
    #    从这里开始就错位了，实测只有约 17% 的张量（基本是最前面的 stem 层）能
    #    按 名称+形状 对上，其余绝大部分层仍然是随机初始化——这也是之前"直接从零
    #    训练不稳定"的根本原因：预训练权重本来就没怎么生效，问题出在超参上。
    custom_yaml = 'ultralytics/cfg/models/26/yolo26-micro.yaml'
    local_weights_path = './weights/yolo26n-cls.pt'

    print(f"加载自定义结构: {custom_yaml}，并挂载可对齐的预训练权重: {local_weights_path}")
    model = YOLO(custom_yaml).load(local_weights_path)

    # 2. 核心训练配置
    results = model.train(
        data='datasets/rafdb_yolo',  # 我们刚刚精心整理的数据集根目录
        task='classify',
        epochs=150,  # 大部分层是随机初始化，需要更多轮数才能充分收敛
        patience=30,  # 早停：连续 30 轮验证集无提升就停止，防止过拟合
        imgsz=128,  # 图像尺寸：分类任务的标准输入大小
        batch=64,  # 批次大小：如果你的电脑显存/内存够大，可以开到 64 或 128
        cache="ram", #训练之前先将图片从硬盘中提前读取到内存中
        workers=8,  # 数据加载线程数，优化数据读取速度
        device='',  # 留空代表自动选择 (优先 CUDA，其次 MPS，最后 CPU)
        # project='runs/classify',  # 训练日志和权重的保存主目录
        name='fer_micro_cls',  # 本次实验的名称
        pretrained=False,  # 预训练权重已经通过上面的 .load() 手动挂载，这里不再触发自动下载

        # --- 稳定训练的超参 (缓解自定义结构大量随机初始化带来的震荡) ---
        lr0=0.001,  # 降低初始学习率，避免训练初期梯度过大导致震荡/发散
        warmup_epochs=5,  # 学习率热身轮数，让优化器先"热身"再全速训练
        cos_lr=True,  # 余弦退火学习率，后期收敛更平稳
        dropout=0.2,  # 分类头 dropout，缓解小模型在小样本类别上过拟合

        # --- 数据增强策略 (对抗网课复杂环境的法宝) ---
        fliplr=0.5,  # 50% 概率水平翻转 (增强面部左右对称泛化)
        hsv_h=0.015,  # 色调微调 (对抗不同房间的光线变化)
        hsv_s=0.7,  # 饱和度微调
        hsv_v=0.4  # 亮度微调
    )

    print("🎉 训练任务执行完毕！")
    print("最佳权重文件 (best.pt) 已保存在: runs/classify/fer_micro_cls/weights/ 目录下。")


if __name__ == '__main__':
    # 在 Windows/macOS 下，多进程训练必须放在 __main__ 保护块中
    train_emotion_model()