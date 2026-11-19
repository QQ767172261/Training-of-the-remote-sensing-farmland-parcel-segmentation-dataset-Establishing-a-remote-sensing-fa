遥感农田地块分割数据集的训练 建立遥感农田地块语义分割系统 如何应用于 Sentinel-2、高分影像、吉林一号等多源遥感数据 实现遥感图像中的农田地块语义分割任务
@[toc]
以下为代码示例，仅供参考学习。

1：sentinel2卫星，224×224尺寸，共1991对图像2：超高分辨率影像，500×500尺寸，1200对图像3：国内东北一号地块分割，共16张大tif遥感农田分割数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9e1ce65a9054f8bac2d1d0c7c6e90c2.png)


遥感农田地块分割合集 数据1：sentinel2卫星，224×224尺寸，共1991对图像。 数据2：超高分辨率影像，500×500尺寸，1200对图像。 数据3：国内吉林一号地块分割，共16张大尺幅tif影像shp标注文件![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f9bc6856675e4e70a48ddd430a809dce.png)


**遥感农田地块分割数据集合集**，包含以下三类数据：

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f07b1e67f5984c5984e6b154e27e9d9b.png)

## 📊 数据集总览
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/17eedd2d2d774b04851d015d7e66bea0.png)

| 类型 | 图像尺寸 | 数量 | 描述 |
|------|----------|------|------|
| **数据1：Sentinel-2 卫星图像** | 224×224 | 1991对 | RGB+多光谱合成图 + 对应mask（二值或类别） |
| **数据2：超高分辨率影像** | 500×500 | 1200对 | 高清RGB图像 + 地块mask |
| **数据3：吉林一号大尺寸遥感图像** | 大tif（如5000x5000） | 16张 | TIF遥感图像 + Shapefile标注文件 |

> ✅ 所有数据可用于**遥感图像中的农田地块语义分割任务**。

---

## 🧠 任务目标

使用深度学习模型（如U-Net、DeepLabV3+、PSPNet等）进行：
- **遥感图像中农田地块的像素级分割**
- 支持多种分辨率输入
- 可扩展至全国/区域尺度农田提取

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db3c6d398bde47a18aa6e7315848f422.png)

## 🧱 数据结构建议（PyTorch兼容）

```
farmland_segmentation_dataset/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

对于数据3的大tif图像，需先裁剪为训练可用的小patch。

---

## 🧪 数据预处理脚本（适用于所有数据）

### 🔹 裁剪大尺寸遥感图像（如数据3）

```python
import os
from PIL import Image
import numpy as np
import rasterio
from shapely.geometry import shape
import geopandas as gpd
import json
import cv2
from tqdm import tqdm

def crop_tif_to_patches(tif_path, output_dir, patch_size=512):
    with rasterio.open(tif_path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)  # HWC
        height, width = img.shape[:2]

        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    out_path = os.path.join(output_dir, f"patch_{i}_{j}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

# 示例调用
crop_tif_to_patches("data/jilin1.tif", "patches/images/", patch_size=512)
```

---

### 🔹 Shapefile 转 Mask（适用于数据3）

```python
import rasterio
from shapely.geometry import mapping
from rasterio.features import rasterize
import geopandas as gpd
import os

def shp_to_mask(tif_path, shp_path, output_mask_path):
    with rasterio.open(tif_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs

    shapes = gpd.read_file(shp_path).to_crs(crs)
    mask = rasterize(
        ((shape(geom), 1) for geom in shapes.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=rasterio.uint8
    )

    with rasterio.open(
        output_mask_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(mask, 1)

# 示例调用
shp_to_mask("data/jilin1.tif", "data/jilin1.shp", "masks/jilin1_mask.tif")
```

---

## 📦 构建统一的数据加载器（支持多种尺寸）

```python
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np

class FarmlandDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取mask
        mask = cv2.imread(mask_path, 0)

        # 二值化处理
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()  # [C, H, W]
```

---

## 🧠 推荐模型架构

| 模型 | 特点 | 推荐理由 |
|------|------|----------|
| **U-Net** | 编码器-解码器 + 跳跃连接 | 小样本友好，泛化能力强 |
| **DeepLabV3+ (ResNet50)** | 空洞卷积 + ASPP模块 | 更好处理大尺度变化 |
| **PSPNet (ResNet50)** | 金字塔池化 | 上下文信息更强 |
| **HRNet** | 多分辨率并行 | 保留细节信息强 |

📌 **推荐模型：DeepLabV3+**

- 支持多尺度特征融合
- 对高分辨率图像适应性强
- PyTorch官方支持 `torchvision.models.segmentation.deeplabv3_resnet50`

---

## 🏋️‍♂️ 使用 DeepLabV3+ 训练代码（PyTorch）

```python
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 初始化模型
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 数据增强与加载
transform = A.Compose([
    A.Resize(height=512, width=512),
    A.ToFloat(max_value=255),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_dataset = FarmlandDataset("images/train", "masks/train", transform=transform)
val_dataset = FarmlandDataset("images/val", "masks/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 优化器 & 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 开始训练
for epoch in range(50):  # 50个epoch
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)['out']
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/50], Loss: {total_loss / len(train_loader):.4f}")
```

---

## 🔍 推理与可视化（单图）

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    images, masks = next(iter(val_loader))
    outputs = model(images.to(device))['out'].cpu().sigmoid().numpy()

    for i in range(len(outputs)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(masks[i][0].cpu().numpy(), cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(outputs[i][0] > 0.5, cmap='gray')

        plt.show()
```

---

## 📊 性能评估指标（IoU, Dice）

```python
from sklearn.metrics import jaccard_score, f1_score

def calculate_metrics(preds, targets):
    preds = preds.flatten()
    targets = targets.flatten()
    iou = jaccard_score(targets, preds)
    dice = f1_score(targets, preds)
    return {"IoU": iou, "Dice": dice}
```

---

## 📦 模型导出（ONNX）

```python
dummy_input = torch.randn(1, 3, 512, 512).to(device)
model.eval()
torch.onnx.export(model, dummy_input, "deeplab_farmland.onnx", export_params=True)
```

---

## 📌 多数据源融合训练策略

你可以将不同数据源合并训练：

| 数据类型 | 尺寸 | 是否增强 | 建议 |
|---------|------|-----------|------|
| Sentinel-2 | 224×224 | 是 | 数据丰富，适合基础训练 |
| 超高分辨率影像 | 500×500 | 是 | 提升精度，注意裁剪 |
| 吉林一号 | 5000×5000 → 512×512 patches | 否（已裁剪） | 用于验证和测试 |

✅ 建议统一缩放至 512×512，或使用多尺度训练策略（`A.RandomScale()`）

---

## ✅ 小结

 **遥感农田地块语义分割系统训练与部署方案**，用于 Sentinel-2、高分影像、吉林一号等多源遥感数据。以下是关键点总结：

| 步骤 | 内容 |
|------|------|
| 数据结构 | 按图像/mask组织 |
| 模型 | 使用 DeepLabV3+ |
| 损失函数 | BCEWithLogitsLoss |
| 推理 | 支持可视化预测结果 |
| 评估 | 提供 IoU 和 Dice Score |
| 部署 | 支持 ONNX 导出 |

---


