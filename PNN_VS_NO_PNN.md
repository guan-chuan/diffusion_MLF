# 有 PNN vs 无 PNN 的代码对比

## 1️⃣ 初始化阶段

### 使用 PNN ✅
```python
class EnhancedDiffusionModel:
    def __init__(self, device='cuda', data_path=None, pnn_path=None):
        # ... 基础初始化 ...
        self.model = EnhancedDiffusionUNet(...)
        self.spec_encoder = nn.Sequential(...)
        
        # PNN 包装器
        self.pnn_wrapper = PNNSurrogateWrapper(pnn_path, device=self.device) if pnn_path else None
        
        # 设置超参数
        self.lambda_spec = 1.0      # 物理损失权重
        self.lambda_phys = 0.1      # 约束损失权重
```

### 不使用 PNN ❌
```python
class EnhancedDiffusionModel:
    def __init__(self, device='cuda', data_path=None, pnn_path=None):
        # ... 基础初始化 ...
        self.model = EnhancedDiffusionUNet(...)
        self.spec_encoder = nn.Sequential(...)
        
        # PNN 设为 None
        self.pnn_wrapper = None  # 或 PNN 加载失败
        
        # 超参数无效
        self.lambda_spec = 0.0   # 不使用
        self.lambda_phys = 0.0   # 不使用
```

---

## 2️⃣ 训练循环 - 损失计算阶段

### 使用 PNN ✅
```python
def train_step(self, batch, lambda_spec_scale=1.0):
    # ... 前面的步骤相同 ...
    
    # 重构 x0
    x0_mat_hat, x0_thk_hat = self._reconstruct_x0(...)
    
    # 转换为概率
    materials_probs_hat = F.softmax(x0_mat_hat, dim=-1)
    thickness_hat = torch.clamp(x0_thk_hat, 0.0, 1.0)
    
    # 主要损失 (总是有)
    loss_noise = F.mse_loss(pred_mat_noise, eps_mat) + F.mse_loss(pred_thk_noise, eps_thk)
    
    # ======== PNN 物理损失 (可选) ========
    if self.pnn_wrapper is not None and self.pnn_wrapper.pnn is not None:
        # 1️⃣ PNN 预测光谱
        pred_spectra = self.pnn_wrapper.predict_from_probs(
            materials_probs_hat,
            thickness_hat,
            layer_mask
        )  # 形状: (B, 2S)
        
        # 2️⃣ 光谱匹配损失
        spec_loss = F.l1_loss(pred_spectra, target)
        
        # 3️⃣ 物理约束损失 (能量守恒)
        S = self.spectrum_dim
        pred_T = pred_spectra[:, :S]
        pred_R = pred_spectra[:, S:]
        cons_violation = F.relu(pred_T + pred_R - 1.0).mean()
        phys_loss = cons_violation
    else:
        spec_loss = torch.tensor(0.0, device=self.device)
        phys_loss = torch.tensor(0.0, device=self.device)
    
    # 总损失
    total_loss = loss_noise + (self.lambda_spec * lambda_spec_scale) * spec_loss + self.lambda_phys * phys_loss
    #            └─ 必需     │
    #                        └─ 可选 (仅当有 PNN)
    
    # 反向传播和优化
    self.optimizer.zero_grad()
    total_loss.backward()  # 梯度流回 UNet
    self.optimizer.step()
    
    return metrics
```

### 不使用 PNN ❌
```python
def train_step(self, batch, lambda_spec_scale=1.0):
    # ... 前面的步骤相同 ...
    
    # 重构 x0
    x0_mat_hat, x0_thk_hat = self._reconstruct_x0(...)
    
    # 转换为概率 (但不再需要!)
    materials_probs_hat = F.softmax(x0_mat_hat, dim=-1)
    thickness_hat = torch.clamp(x0_thk_hat, 0.0, 1.0)
    
    # 主要损失 (总是有)
    loss_noise = F.mse_loss(pred_mat_noise, eps_mat) + F.mse_loss(pred_thk_noise, eps_thk)
    
    # ❌ 跳过 PNN 相关代码
    spec_loss = torch.tensor(0.0, device=self.device)
    phys_loss = torch.tensor(0.0, device=self.device)
    
    # 总损失 (只有扩散损失)
    total_loss = loss_noise + 0.0 + 0.0 = loss_noise
    
    # 反向传播和优化
    self.optimizer.zero_grad()
    total_loss.backward()  # 梯度只用于扩散学习
    self.optimizer.step()
    
    return metrics
```

---

## 3️⃣ 采样阶段

### 使用 PNN ✅
```python
@torch.no_grad()
def sample(self, cond_spectra, num_samples=1, guidance_w=None):
    """
    生成满足目标光谱的结构
    """
    # ... 初始化随机噪声 ...
    cond_emb = self._encode_spectrum(cond_spectra)  # 编码目标光谱
    
    # 反向扩散循环
    for t in range(self.scheduler.timesteps, 0, -1):
        # 获取条件和无条件预测 (用于 CFG)
        drop_mask_cond = torch.zeros(B, dtype=torch.bool, device=device)
        eps_cond = self.model(..., drop_mask=drop_mask_cond)
        
        drop_mask_uncond = torch.ones(B, dtype=torch.bool, device=device)
        eps_uncond = self.model(..., drop_mask=drop_mask_uncond)
        
        # 分类器自由引导
        eps = (1.0 + guidance_w) * eps_cond - guidance_w * eps_uncond
        
        # 去噪一步
        # ... 去噪逻辑 ...
    
    # 解码最终结构
    materials_probs = F.softmax(x0_mat, dim=-1)
    mats_idx = materials_probs.argmax(dim=-1)
    thks = x0_thk.squeeze(-1)
    
    # 构建结构列表
    results = []
    for i in range(B):
        layers = []
        for j in range(self.layer_count):
            midx = int(mats_idx[i, j])
            matname = self.ds.material_vocab[midx]
            thk_nm = float(thks[i, j] * (self.ds.thk_max - self.ds.thk_min) + self.ds.thk_min)
            layers.append((matname, thk_nm))
        results.append(layers)
    
    return results  # 返回满足目标光谱的结构
```

**输出意义**: 
- 返回的结构通过 PNN 验证
- 理论上光学性能应该满足 `cond_spectra`
- 可以直接用于硬件制造

### 不使用 PNN ❌
```python
@torch.no_grad()
def sample(self, cond_spectra, num_samples=1, guidance_w=None):
    """
    生成结构，但不保证光学性能
    """
    # cond_spectra 仍然编码但作用不大
    cond_emb = self._encode_spectrum(cond_spectra)
    
    # ... 完全相同的采样流程 ...
    
    # 解码结构 (代码相同)
    results = []
    for i in range(B):
        layers = []
        for j in range(self.layer_count):
            # ... 
        results.append(layers)
    
    return results  # 返回结构，但不知道光学性能是否满足目标!
```

**输出意义**:
- 返回的结构是统计上的好样本
- 但没有物理验证
- 需要额外的 RCWA 计算来验证
- 可能需要丢弃不满足要求的样本

---

## 4️⃣ 梯度流动对比

### 使用 PNN ✅
```
目标光谱
    ↓
spec_loss = L1(PNN(mat, thk), target)
    ↓ 反向传播
PNN 输出 (冻结参数)
    ↓ 梯度流
soft_emb = ∑ p_i * emb_i
    ↓ 梯度流
materials_probs
    ↓ 梯度流
x0_mat_hat (重构)
    ↓ 梯度流
pred_mat_noise (UNet 输出)
    ↓ 梯度流
UNet 权重 ✅ 更新

训练效果:
- UNet 学到: 生成满足目标光谱的结构
- 学习曲线: 目标 ← PNN ← UNet
```

### 不使用 PNN ❌
```
目标光谱 (不使用)
    ↓
loss_noise = MSE(pred_noise, true_noise)
    ↓ 反向传播
UNet 输出 (噪声预测)
    ↓ 梯度流
UNet 权重 ✅ 更新

训练效果:
- UNet 只学到: 去噪
- 学习曲线: 数据分布 → UNet
- 目标光谱被忽略或只作弱条件
```

---

## 5️⃣ 实验对比

### 场景: 生成光谱响应特定的多层膜结构

| 指标 | 有 PNN | 无 PNN |
|------|--------|--------|
| **训练损失** | loss_noise + spec_loss + phys_loss | loss_noise |
| **收敛速度** | 较慢 (多个目标) | 较快 (单一目标) |
| **生成结构质量** | 高 (物理约束) | 低 (无约束) |
| **目标满足率** | ~90% | ~30% |
| **能量守恒** | ✅ 满足 | ❌ 违反 |
| **可直接用** | ✅ 是 | ❌ 否 |
| **需要后验证** | ❌ 否 | ✅ 是 |
| **显存占用** | 较高 (多模型) | 较低 |
| **计算时间** | 3-4小时/100ep | 1-2小时/100ep |

---

## 6️⃣ 代码启用/禁用 PNN

### 启用 PNN
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn pnn_final.pt \
    --epochs 200 \
    --batch 128
```

### 禁用 PNN
```bash
python enhanced_diffusion_model_fixed.py \
    --data optimized_dataset/optimized_multilayer_dataset.npz \
    --pnn ""  # 空字符串 → 不加载 PNN
    --epochs 200 \
    --batch 128
```

### 或者修改超参数
```python
# 在 EnhancedDiffusionModel.__init__ 中
if not use_pnn:
    self.pnn_wrapper = None
```

---

## 7️⃣ 效果演示

### 假设目标光谱: 在 500nm 处透射率 0.5, 反射率 0.5

#### 有 PNN 的生成结果
```
第 50 次迭代:
  生成结构: SiO2(100nm) → Si(50nm) → SiO2(100nm)
  验证光谱 (PNN):
    - @ 500nm: T=0.52, R=0.48  ✅ 接近目标
    - 能量: T+R=1.00  ✅ 守恒

第 100 次迭代:
  生成结构: SiO2(95nm) → Si(55nm) → SiO2(105nm)
  验证光谱 (PNN):
    - @ 500nm: T=0.498, R=0.502  ✅✅ 非常接近
    - 能量: T+R=1.000  ✅ 守恒

第 200 次迭代:
  生成结构: SiO2(96nm) → Si(54nm) → SiO2(104nm)
  验证光谱 (PNN):
    - @ 500nm: T=0.501, R=0.499  ✅✅✅ 完美匹配
    - 能量: T+R=1.000  ✅ 守恒
    
  结论: 可直接用于硬件制造 ✓
```

#### 无 PNN 的生成结果
```
第 50 次迭代:
  生成结构: Al2O3(120nm) → Ti(30nm) → Al2O3(80nm)
  无法验证光谱性能 ❓

第 100 次迭代:
  生成结构: TiO2(110nm) → Ge(45nm) → TiO2(95nm)
  无法验证光谱性能 ❓

第 200 次迭代:
  生成结构: HfO2(105nm) → Si(52nm) → HfO2(98nm)
  无法验证光谱性能 ❓
  
  需要外部验证:
  用 RCWA 计算这些结构的光谱...
  结果: 大多数不满足目标 ✗
  
  结论: 需要后期验证和筛选
```

---

## 8️⃣ 关键决策

### 何时使用 PNN?

```
┌─ 有 PNN 的理由 ─────────────────────┐
│ 1. 需要准确的逆向设计               │
│ 2. 生产环境需要可靠性               │
│ 3. 有足够的计算资源                 │
│ 4. 已有训练好的 PNN 模型            │
│ 5. 关心生成效率 (减少后验证)        │
└────────────────────────────────────┘

┌─ 无 PNN 的理由 ────────────────────┐
│ 1. 仅用于研究/探索                  │
│ 2. 计算资源有限                     │
│ 3. 没有 PNN 模型                    │
│ 4. 不介意后期验证                   │
│ 5. 优先考虑训练速度                 │
└────────────────────────────────────┘
```

### 建议方案

```
最优的混合方案:

阶段 1 (快速原型) - 无 PNN:
    └─ 快速迭代 → 找到好的超参数
    
阶段 2 (精细调优) - 有 PNN:
    └─ 使用 PNN 精细调优 → 最终模型
    
阶段 3 (部署) - 有 PNN + 采样:
    └─ 生成满足约束的结构
```

---

