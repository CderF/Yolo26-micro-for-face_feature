import torch
from ultralytics import YOLO

print("🚀 开始读取 yolo26-micro.yaml，尝试动态构建网络...")
# 这一步极其致命。Ultralytics 引擎会逐行读取你的 YAML。
# 如果我们在 Concat 层的参数或维度算错了，代码在这里就会瞬间崩溃，报 Shape Mismatch！
model = YOLO("ultralytics/cfg/models/26/yolo26-micro.yaml")

print("✅ 网络物理结构拼接成功！开始构造 Mock 张量...")
# 我们伪造一张批次为 1，通道为 3（RGB），尺寸为 640x640 的纯黑“假照片”
dummy_tensor = torch.zeros((1, 3, 640, 640))

print("开始进行前向传播 (Forward Pass)...")
# 让这个假张量流经我们刚刚设计的 Backbone -> Neck -> Head
# 设置 verbose=False 是为了不打印花哨的推理框信息，我们只看能不能跑通
results = model.predict(dummy_tensor, imgsz=640, verbose=False)

print("成功通过前向传播")