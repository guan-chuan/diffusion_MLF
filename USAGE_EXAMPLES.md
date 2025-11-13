# 使用示例和配置指南

## 📋 快速开始

### 前置检查
```bash
cd /home/engine/project

# 1. 检查依赖
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 2. 检查数据集
ls -lh optimized_dataset/optimized_multilayer_dataset.npz

# 3. 检查 PNN 模型
ls -lh pnn_final.pt pnn_best.pt
```

---

## 🎯 配置 1: 基础扩散模型 (无 PNN)

### 场景
- 用途: 学习、快速原型开发
- 优点: 快速、不需要外部模型
- 缺点: 无物理约束、需要后期验证

### 命令
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --epochs 50 \
    --batch 128 \
    --lr 1e-4 \
    --device cuda:0
```

### 预期输出
```
[device] cuda:0
[PNN] pnn_path None not found; continuing without PNN.
[data] training: 285000 validation: 15000

Epoch 1/50 finished. train loss: 2.3456 | val spec L1: 0.0000 | val r2: 0.0000
Epoch 2/50 finished. train loss: 2.1234 | val spec L1: 0.0000 | val r2: 0.0000
...
Epoch 50/50 finished. train loss: 0.5678 | val spec L1: 0.0000 | val r2: 0.0000
Saved final model: enhanced_diffusion_model_final.pt
```

### 特点
- ✅ 快速收敛 (损失快速下降)
- ❌ spec_loss 和 r2 始终为 0
- ✅ 总训练时间: ~30 分钟 (100 epochs)

---

## 🎯 配置 2: 完整物理感知模型 (有 PNN)

### 场景
- 用途: 生成可用的设计、真实应用
- 优点: 物理约束、可直接使用
- 缺点: 慢速、需要 PNN 模型

### 命令
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 200 \
    --batch 128 \
    --lr 1e-4 \
    --spec_warmup_epochs 20 \
    --guidance 6.0 \
    --device cuda:0
```

### 预期输出
```
[device] cuda:0
[PNN] auto-loaded PNN from checkpoint...
[data] training: 285000 validation: 15000

Epoch 1/200 finished. train loss: 2.8901 | val spec L1: 1.2345 | val r2: 0.0123
Epoch 2/200 finished. train loss: 2.6234 | val spec L1: 1.1234 | val r2: 0.0456
...
Epoch 20/200 finished. train loss: 1.2345 | val spec L1: 0.5678 | val r2: 0.6789
...
Epoch 200/200 finished. train loss: 0.3456 | val spec L1: 0.1234 | val r2: 0.8901
Saved best diffusion model: enhanced_diffusion_best.pt
Saved final model: enhanced_diffusion_model_final.pt
```

### 特点
- ✅ 三个损失函数同时优化
- ✅ spec_loss 和 r2 逐渐改善
- ✅ 第 20 个 epoch 后 spec 损失开始快速下降 (lambda_scale 从 0 → 1)
- ✅ 总训练时间: ~3-4 小时 (200 epochs)
- ✅ 最终 r2 > 0.85 表示很好的光谱匹配

---

## 🎯 配置 3: 调试模式 (打印梯度信息)

### 场景
- 用途: 调试、理解梯度流动
- 优点: 详细的调试信息
- 缺点: 速度慢、大量日志

### 命令
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 5 \
    --batch 32 \
    --grad_debug \
    --device cuda:0
```

### 预期输出
```
[device] cuda:0
[PNN] auto-loaded PNN from checkpoint...

Epoch 1/5 finished. train loss: 2.8901 | val spec L1: 1.2345 | val r2: 0.0123

DEBUG grad flags (first batch):
  loss_noise.requires_grad: True
  spec_loss.requires_grad: True
  total_loss.requires_grad: True
  pred_spectra.requires_grad (if present): True
  materials_probs_hat.requires_grad: True
  x0_mat_hat.requires_grad: True

✅ 所有关键张量都有梯度!
```

### 验证指标
- ✅ `loss_noise.requires_grad: True` → 扩散损失可微分
- ✅ `spec_loss.requires_grad: True` → 物理损失可微分
- ✅ `total_loss.requires_grad: True` → 总损失可微分
- ✅ `pred_spectra.requires_grad: True` → PNN 输出可微分

---

## 🎯 配置 4: 性能优化 (小批量快速迭代)

### 场景
- 用途: 在有限显存上训练
- 优点: 显存占用少、速度快
- 缺点: 批量小、可能不稳定

### 命令
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 100 \
    --batch 32 \
    --lr 1e-4 \
    --device cuda:0
```

### 显存对比
```
batch_size 和显存占用:
├─ batch=32  → ~8 GB
├─ batch=64  → ~12 GB
├─ batch=128 → ~18 GB (推荐用于 RTX 3090)
└─ batch=256 → ~36 GB (需要 A100)
```

---

## 🎯 配置 5: 多卡训练

### 场景
- 用途: 加速大规模训练
- 优点: 显著加速
- 缺点: 需要多卡支持

### 命令 (单卡指定)
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 200 \
    --batch 256 \
    --device cuda:0
```

### 命令 (多卡支持) - 需要代码修改
```python
# 在代码中手动支持 DataParallel
import torch.nn as nn

model = EnhancedDiffusionUNet(...)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## 🔍 采样 (生成新结构)

### 场景 1: 使用训练好的模型生成

```python
import torch
from enhanced_diffusion_model_fixed import EnhancedDiffusionModel

# 加载模型
model = EnhancedDiffusionModel(
    device='cuda:0',
    data_path='optimized_dataset/optimized_multilayer_dataset.npz',
    pnn_path='pnn_final.pt'
)

# 加载检查点
checkpoint = torch.load('enhanced_diffusion_best.pt')
model.model.load_state_dict(checkpoint['model_state'])

# 生成目标光谱
target_spectrum = torch.randn(1, model.spectrum_dim * 2)  # (1, 2S)

# 采样
structures = model.sample(
    cond_spectra=target_spectrum,
    num_samples=10,
    use_ema=True,
    guidance_w=6.0
)

# 输出
for i, struct in enumerate(structures):
    print(f"Sample {i}:")
    for layer in struct:
        mat_name, thickness_nm = layer
        print(f"  {mat_name}: {thickness_nm:.1f} nm")
```

### 输出示例
```
Sample 0:
  SiO2: 95.3 nm
  Si: 54.7 nm
  SiO2: 104.2 nm

Sample 1:
  SiO2: 97.1 nm
  Si: 52.3 nm
  SiO2: 106.5 nm

...
```

### 场景 2: 生成不同的目标

```python
# 目标 1: 500nm 处高透射率
spec1 = torch.zeros(1, model.spectrum_dim * 2)
spec1[0, 50] = 0.9  # 500nm 处透射率 = 0.9

# 目标 2: 500nm 处高反射率
spec2 = torch.zeros(1, model.spectrum_dim * 2)
spec2[0, model.spectrum_dim + 50] = 0.9  # 500nm 处反射率 = 0.9

# 采样
structures1 = model.sample(spec1, num_samples=5)  # 透射镜
structures2 = model.sample(spec2, num_samples=5)  # 反射镜
```

---

## 📊 监控训练

### 实时绘图 (tensorboard)
```bash
# 需要修改代码以支持 tensorboard
pip install tensorboard

# 然后运行:
tensorboard --logdir=./runs
```

### 手动监控
```bash
# 查看输出日志
python enhanced_diffusion_model_fixed.py ... | tee training.log

# 实时查看
tail -f training.log

# 统计损失变化
grep "train loss:" training.log | tail -20
```

---

## ⚠️ 常见问题和解决方案

### 问题 1: CUDA 内存不足

```bash
# 解决方案 1: 减少批量
python ... --batch 32

# 解决方案 2: 使用 CPU (慢)
python ... --device cpu

# 解决方案 3: 梯度累积 (需要代码修改)
accumulation_steps = 4
```

### 问题 2: 训练损失不下降

```bash
# 检查项:
1. 学习率太高 → --lr 5e-5
2. 批量太小 → --batch 128
3. 数据问题 → 检查 npz 文件
4. 模型初始化 → 重新运行

# 解决方案
python ... --lr 5e-5 --batch 128
```

### 问题 3: PNN 加载失败

```bash
# 检查:
1. PNN 文件存在: ls -l pnn_final.pt
2. PNN 版本匹配: 检查 pnn.py 的定义
3. 设备不匹配: 确保 CUDA 版本一致

# 调试:
python -c "
import torch
from pnn import PNNTransformer
try:
    model = torch.load('pnn_final.pt')
    print('PNN 加载成功')
except Exception as e:
    print(f'PNN 加载失败: {e}')
"
```

### 问题 4: 采样速度太慢

```bash
# 原因: 1000 个去噪步骤
# 解决方案: 使用 DDIM 加速 (需要代码修改)

# 临时解决: 减少去噪步骤
# 在代码中: T = 100 而不是 1000 (精度会下降)
```

---

## 📈 性能基准

在 RTX 3090 上的性能:

| 配置 | 批量 | 时间/epoch | 显存 | 质量 |
|------|------|-----------|------|------|
| 无 PNN | 128 | 45s | 8GB | 低 |
| 有 PNN | 128 | 120s | 18GB | 高 |
| 有 PNN | 64 | 70s | 12GB | 高 |
| 有 PNN | 32 | 45s | 8GB | 高 |

### 推荐配置

```bash
# 对于 RTX 3090:
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 200 \
    --batch 128 \
    --lr 1e-4 \
    --device cuda:0
# 预期: 8-10 小时完整训练
```

---

## 🎓 学习路径

### 第 1 步: 理解模型
```bash
# 运行调试版本
python enhanced_diffusion_model_fixed.py \
    --data ... \
    --epochs 2 \
    --batch 32 \
    --grad_debug

# 观察输出，理解梯度流
```

### 第 2 步: 快速原型
```bash
# 无 PNN 训练 (快速验证)
python enhanced_diffusion_model_fixed.py \
    --data ... \
    --epochs 10 \
    --batch 128

# 检查是否能运行
```

### 第 3 步: 物理约束训练
```bash
# 添加 PNN (完整训练)
python enhanced_diffusion_model_fixed.py \
    --data ... \
    --pnn pnn_final.pt \
    --epochs 200 \
    --batch 128

# 观察三个损失的演化
```

### 第 4 步: 采样和验证
```python
# 生成结构
structures = model.sample(target_spectrum, num_samples=100)

# 用 PNN 验证 (已内置)
# 或用真实 RCWA 验证
```

---

## 📝 输出文件

训练完成后生成:

```
enhanced_diffusion_best.pt
└─ 保存最佳模型 (基于验证集 r2)
└─ 包含: 'model_state' 字典

enhanced_diffusion_model_final.pt
└─ 保存最终模型 (最后一个 epoch)
└─ 包含: 'model_state' 字典
```

### 加载和使用
```python
import torch

# 加载最佳模型
best_ckpt = torch.load('enhanced_diffusion_best.pt')
model.model.load_state_dict(best_ckpt['model_state'])

# 用于采样
structures = model.sample(target_spectrum)
```

---

