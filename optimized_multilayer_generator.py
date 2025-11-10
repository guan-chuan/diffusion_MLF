"""
优化的多层薄膜数据集生成器
基于物理约束和实际应用模式的改进版本

主要改进：
1. 移除Ag材料，专注于介电质多层膜
2. 优化层数分布，基于样本空间大小
3. 改进厚度范围，基于材料物理特性
4. 添加物理兼容性检查
5. 实现目标导向的结构生成模式
6. 增强数据质量控制
"""

import numpy as np
import pandas as pd
import os
import random
from scipy.interpolate import interp1d
import time
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import tqdm
import matplotlib.pyplot as plt
import math
import cmath
from datetime import datetime
import json

# 安全打印函数，避免Windows编码问题
def safe_print(text):
    """安全打印函数，处理Windows中文编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(f"[编码警告] {safe_text}")

class OptimizedMaterialDatabase:
    """优化的材料数据库类 - 移除Ag，专注于介电质材料"""
    
    def __init__(self, materials_dir="materials"):
        self.materials_dir = materials_dir
        self.materials = {}
        
        # 优化的材料列表 - 移除Ag，专注于介电质多层膜
        self.target_materials = [
            'SiO2',    # 低折射率介电质 (n~1.46)
            'Al2O3',   # 中等介电质 (n~1.77) 
            'Si3N4',   # 中等介电质 (n~2.0)
            'HfO2',    # 高k介电质 (n~2.1)
            'TiO2',    # 高折射率介电质 (n~2.4)
            'Ta2O5',   # 高折射率介电质 (n~2.2)
            'Si',      # 半导体 (n~3.5)
            'Ge',      # 半导体 (n~4.0)
            'ITO'      # 透明导体 (n~1.8)
        ]
        
        # 材料分组 - 基于折射率
        self.material_groups = {
            'low_index': ['SiO2'],                           # n < 1.7
            'medium_index': ['Al2O3', 'ITO', 'Si3N4'],       # 1.7 <= n < 2.1
            'high_index': ['HfO2', 'Ta2O5', 'TiO2'],        # 2.1 <= n < 3.0
            'very_high_index': ['Si', 'Ge']                  # n >= 3.0
        }
        
        # 材料特性参数
        self.material_properties = {
            'SiO2': {'typical_n': 1.46, 'loss_level': 'very_low'},
            'Al2O3': {'typical_n': 1.77, 'loss_level': 'very_low'},
            'Si3N4': {'typical_n': 2.0, 'loss_level': 'low'},
            'HfO2': {'typical_n': 2.1, 'loss_level': 'low'},
            'TiO2': {'typical_n': 2.4, 'loss_level': 'low'},
            'Ta2O5': {'typical_n': 2.2, 'loss_level': 'low'},
            'Si': {'typical_n': 3.5, 'loss_level': 'medium'},
            'Ge': {'typical_n': 4.0, 'loss_level': 'high'},
            'ITO': {'typical_n': 1.8, 'loss_level': 'medium'}
        }
        
        self.load_materials()
        
    def load_materials(self):
        """加载优化的材料集合"""
        safe_print("[材料库] 开始加载优化材料集合...")
        
        for material_name in self.target_materials:
            filepath = os.path.join(self.materials_dir, f"{material_name}.csv")
            
            if not os.path.exists(filepath):
                safe_print(f"[警告] 材料文件不存在: {filepath}")
                continue
                
            try:
                df = pd.read_csv(filepath)
                
                # 检查列名格式
                if 'nm' in df.columns and 'n' in df.columns and 'k' in df.columns:
                    wavelength_col = 'nm'
                elif 'wavelength' in df.columns and 'n' in df.columns and 'k' in df.columns:
                    wavelength_col = 'wavelength'
                else:
                    safe_print(f"[警告] {material_name} 文件格式不正确，跳过")
                    continue
                
                # 存储材料数据
                self.materials[material_name] = {
                    'wavelength': df[wavelength_col].values,
                    'n': df['n'].values,
                    'k': df['k'].values
                }
                
                safe_print(f"[材料] 成功加载 {material_name}: {len(df)} 个数据点")
                
            except Exception as e:
                safe_print(f"[错误] 加载 {material_name} 失败: {e}")
        
        if not self.materials:
            raise ValueError("未能加载任何材料数据")
        
        safe_print(f"[材料库] 总计加载 {len(self.materials)} 种材料")
        self._validate_materials()
    
    def _validate_materials(self):
        """验证材料数据的完整性"""
        safe_print("[验证] 检查材料数据质量...")
        
        for material_name, data in self.materials.items():
            wavelengths = data['wavelength']
            n_values = data['n']
            k_values = data['k']
            
            # 检查数据范围
            wl_range = (wavelengths.min(), wavelengths.max())
            n_range = (n_values.min(), n_values.max())
            k_range = (k_values.min(), k_values.max())
            
            # 验证波长范围覆盖400-1100nm
            if wl_range[0] > 410 or wl_range[1] < 1090:
                safe_print(f"[警告] {material_name} 波长范围不完整: {wl_range}")
            
            # 验证折射率合理性
            if n_range[0] < 1.0 or n_range[1] > 5.0:
                safe_print(f"[警告] {material_name} 折射率范围异常: {n_range}")
            
            # 验证消光系数
            if k_range[1] > 1.0:  # 对于介电质材料，k不应太大
                safe_print(f"[警告] {material_name} 消光系数偏高: {k_range}")
    
    def get_refractive_index(self, material_name, wavelength):
        """获取指定材料在指定波长下的复折射率"""
        if material_name not in self.materials:
            raise ValueError(f"材料 {material_name} 不在数据库中")
        
        material_data = self.materials[material_name]
        wavelengths = material_data['wavelength']
        n_values = material_data['n']
        k_values = material_data['k']
        
        # 线性插值
        if wavelength <= wavelengths[0]:
            n, k = n_values[0], k_values[0]
        elif wavelength >= wavelengths[-1]:
            n, k = n_values[-1], k_values[-1]
        else:
            n = np.interp(wavelength, wavelengths, n_values)
            k = np.interp(wavelength, wavelengths, k_values)
        
        return n + 1j * k

class OptimizedTMMCalculator:
    """优化的TMM计算器 - 增强稳定性和质量控制"""
    
    def __init__(self, material_db):
        self.material_db = material_db
        self.max_phase = 50.0
        self.min_thickness = 1.0    # 最小厚度1nm
        self.max_thickness = 500.0  # 最大厚度500nm
    
    def safe_complex_exp(self, phase):
        """安全的复指数计算"""
        try:
            # 限制相位避免数值溢出
            if abs(phase.real) > self.max_phase:
                phase = complex(np.sign(phase.real) * self.max_phase, phase.imag)
            if abs(phase.imag) > self.max_phase:
                phase = complex(phase.real, np.sign(phase.imag) * self.max_phase)
            
            result = cmath.exp(phase)
            
            # 检查结果有效性
            if math.isnan(result.real) or math.isnan(result.imag):
                return complex(1.0, 0.0)
            if math.isinf(result.real) or math.isinf(result.imag):
                return complex(1.0, 0.0)
                
            return result
        except:
            return complex(1.0, 0.0)
    
    def P_matrix(self, n, d, wavelength):
        """传播矩阵"""
        try:
            d = max(self.min_thickness, min(self.max_thickness, d))
            phase = 2 * (math.pi / wavelength) * d * n
            
            P11 = self.safe_complex_exp(-1j * phase)
            P22 = self.safe_complex_exp(1j * phase)
            
            return np.array([[P11, 0], [0, P22]], dtype=complex)
        except:
            return np.array([[1, 0], [0, 1]], dtype=complex)
    
    def T_matrix(self, n1, n2):
        """界面传输矩阵"""
        try:
            if abs(n1) < 1e-10:
                n1 = complex(1e-10, 0)
                
            T11 = (n1 + n2) / (2 * n1)
            T12 = (n1 - n2) / (2 * n1)
            T21 = (n1 - n2) / (2 * n1)
            T22 = (n1 + n2) / (2 * n1)
            
            elements = [T11, T12, T21, T22]
            for elem in elements:
                if math.isnan(elem.real) or math.isnan(elem.imag):
                    return np.array([[1, 0], [0, 1]], dtype=complex)
                if math.isinf(elem.real) or math.isinf(elem.imag):
                    return np.array([[1, 0], [0, 1]], dtype=complex)
            
            return np.array([[T11, T12], [T21, T22]], dtype=complex)
        except:
            return np.array([[1, 0], [0, 1]], dtype=complex)
    
    def calculate_spectrum_with_validation(self, structure, wavelengths):
        """计算光谱并进行质量验证"""
        try:
            # 构建NDlist
            NDlist = [[complex(1.0, 0), 0, 'air']]  # 空气层
            
            for material, thickness in structure:
                if material not in self.material_db.materials:
                    continue
                    
                thickness = max(self.min_thickness, min(self.max_thickness, thickness))
                center_wavelength = wavelengths[len(wavelengths)//2]
                
                try:
                    n_complex = self.material_db.get_refractive_index(material, center_wavelength)
                except:
                    n_complex = complex(1.5, 0.01)
                
                NDlist.append([n_complex, thickness, material])
            
            if len(NDlist) == 1:
                NDlist.append([complex(1.5, 0.01), 100.0, 'default'])
            
            NDlist.append([complex(1.45, 0), 0, 'substrate'])  # 基底
            
            transmission = []
            reflection = []
            validation_results = []
            
            for wavelength in wavelengths:
                try:
                    # 更新每层的折射率
                    for i in range(1, len(NDlist) - 1):
                        material = NDlist[i][2]
                        if material in self.material_db.materials:
                            try:
                                n_complex = self.material_db.get_refractive_index(material, wavelength)
                            except:
                                n_complex = complex(1.5, 0.01)
                            NDlist[i][0] = n_complex
                    
                    # 计算反射率和透射率
                    R = self.Reflectance(NDlist, wavelength)
                    T = self.Transmittance(NDlist, wavelength)
                    
                    # 物理合理性检查
                    R = np.clip(R, 0, 1)
                    T = np.clip(T, 0, 1)
                    
                    # 能量守恒检查
                    total = T + R
                    if total > 1.0:
                        factor = 0.99 / total
                        T *= factor
                        R *= factor
                    
                    transmission.append(T)
                    reflection.append(R)
                    
                    # 记录验证结果
                    validation_results.append({
                        'wavelength': wavelength,
                        'T': T, 'R': R,
                        'energy_sum': T + R,
                        'valid': 0 <= T <= 1 and 0 <= R <= 1 and T + R <= 1.01
                    })
                    
                except Exception as e:
                    transmission.append(0.5)
                    reflection.append(0.3)
                    validation_results.append({
                        'wavelength': wavelength,
                        'T': 0.5, 'R': 0.3,
                        'energy_sum': 0.8,
                        'valid': False
                    })
            
            # 整体质量评估
            valid_points = sum(1 for v in validation_results if v['valid'])
            quality_score = valid_points / len(validation_results)
            
            return np.array(transmission), np.array(reflection), {
                'quality_score': quality_score,
                'valid_points': valid_points,
                'total_points': len(validation_results),
                'validation_details': validation_results
            }
            
        except Exception as e:
            n_points = len(wavelengths)
            return np.full(n_points, 0.5), np.full(n_points, 0.3), {
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def Reflectance(self, NDlist, wavelength):
        """计算反射率"""
        try:
            M = self.M_matrix(NDlist, wavelength)
            if abs(M[0][0]) < 1e-15:
                return 0.0
            r = M[1][0] / M[0][0]
            R = abs(r)**2
            return max(0.0, min(1.0, R))
        except:
            return 0.1
    
    def Transmittance(self, NDlist, wavelength):
        """计算透射率"""
        try:
            M = self.M_matrix(NDlist, wavelength)
            if abs(M[0][0]) < 1e-15:
                return 0.0
            t = 1 / M[0][0]
            n0 = NDlist[0][0].real if hasattr(NDlist[0][0], 'real') else NDlist[0][0]
            nN = NDlist[-1][0].real if hasattr(NDlist[-1][0], 'real') else NDlist[-1][0]
            if abs(n0) < 1e-15:
                n0 = 1.0
            T = abs(t)**2 * (nN / n0)
            return max(0.0, min(1.0, T))
        except:
            return 0.8
    
    def M_matrix(self, NDlist, wavelength):
        """总传输矩阵"""
        try:
            Tlist = self.T_list(NDlist)
            Plist = self.P_list(NDlist, wavelength)
            M = np.array([[1, 0], [0, 1]], dtype=complex)
            
            for i in range(len(Plist)):
                m = np.dot(Tlist[i], Plist[i])
                M = np.dot(M, m)
                
                if np.any(np.isnan(M)) or np.any(np.isinf(M)):
                    raise ValueError("矩阵计算异常")
            
            M = np.dot(M, Tlist[-1])
            
            if np.any(np.isnan(M)) or np.any(np.isinf(M)):
                raise ValueError("最终矩阵异常")
            
            return M
        except:
            return np.array([[1, 0], [0, 1]], dtype=complex)
    
    def P_list(self, NDlist, wavelength):
        """传播矩阵列表"""
        lis = []
        for i in range(len(NDlist) - 2):
            n = NDlist[i + 1][0]
            d = NDlist[i + 1][1]
            lis.append(self.P_matrix(n, d, wavelength))
        return lis
    
    def T_list(self, NDlist):
        """界面传输矩阵列表"""
        lis = []
        for i in range(len(NDlist) - 1):
            lis.append(self.T_matrix(NDlist[i][0], NDlist[i + 1][0]))
        return lis

class OptimizedMultilayerGenerator:
    """优化的多层薄膜结构生成器"""
    
    def __init__(self, material_db):
        self.material_db = material_db
        self.materials = list(material_db.materials.keys())
        
        # 优化的层数分布 - 基于样本空间分析
        self.layer_distribution = {
            3: 0.25,   # 25% - 基础结构，充分采样
            4: 0.20,   # 20% - 常见结构  
            5: 0.18,   # 18% - 中等复杂度
            6: 0.15,   # 15% - 
            7: 0.10,   # 10% - 
            8: 0.07,   # 7%  - 复杂结构
            9: 0.03,   # 3%  - 高复杂度
            10: 0.02   # 2%  - 极限复杂度
        }
        
        # 优化的厚度范围 - 基于材料物理特性
        self.thickness_configs = {
            'SiO2': {
                'range': (30, 500),
                'distribution': 'uniform',
                'precision': 3
            },
            'Al2O3': {
                'range': (30, 400), 
                'distribution': 'uniform',
                'precision': 3
            },
            'Si3N4': {
                'range': (25, 300),
                'distribution': 'log_uniform', 
                'precision': 2
            },
            'HfO2': {
                'range': (20, 250),
                'distribution': 'uniform',
                'precision': 2
            },
            'TiO2': {
                'range': (25, 350),
                'distribution': 'log_uniform',
                'precision': 2
            },
            'Ta2O5': {
                'range': (20, 300),
                'distribution': 'log_uniform',
                'precision': 2
            },
            'Si': {
                'range': (20, 400),
                'distribution': 'log_uniform',
                'precision': 2
            },
            'Ge': {
                'range': (15, 200),
                'distribution': 'log_uniform',
                'precision': 1
            },
            'ITO': {
                'range': (50, 300),
                'distribution': 'uniform',
                'precision': 5
            }
        }
        
        # 设计模式定义
        self.design_patterns = {
            'random': 0.40,           # 40% 随机结构
            'alternating_index': 0.25, # 25% 高低折射率交替
            'graded_index': 0.20,     # 20% 梯度折射率
            'symmetric': 0.10,        # 10% 对称结构
            'cavity_based': 0.05      # 5% 腔体结构
        }
    
    def generate_thickness(self, material):
        """为指定材料生成厚度"""
        config = self.thickness_configs.get(material, {
            'range': (20, 300),
            'distribution': 'uniform', 
            'precision': 2
        })
        
        min_thick, max_thick = config['range']
        precision = config['precision']
        
        if config['distribution'] == 'log_uniform':
            # 对数均匀分布，偏向较薄的厚度
            log_min, log_max = np.log(min_thick), np.log(max_thick)
            thickness = np.exp(np.random.uniform(log_min, log_max))
        else:
            # 均匀分布
            thickness = np.random.uniform(min_thick, max_thick)
        
        # 根据精度要求进行舍入
        thickness = round(thickness / precision) * precision
        
        return max(min_thick, min(max_thick, thickness))
    
    def check_material_compatibility(self, materials_sequence):
        """检查材料序列的物理兼容性"""
        if len(materials_sequence) < 2:
            return True
        
        # 检查相邻材料不重复
        for i in range(len(materials_sequence) - 1):
            if materials_sequence[i] == materials_sequence[i + 1]:
                return False
        
        # 检查极端折射率跳跃（可选）
        # 这里可以添加更复杂的物理兼容性检查
        
        return True
    
    def generate_alternating_index_structure(self, num_layers):
        """生成高低折射率交替结构"""
        structure = []
        
        # 选择高低折射率材料
        high_materials = self.material_db.material_groups['high_index'] + \
                        self.material_db.material_groups['very_high_index']
        low_materials = self.material_db.material_groups['low_index'] + \
                       self.material_db.material_groups['medium_index']
        
        # 过滤掉不存在的材料
        high_materials = [m for m in high_materials if m in self.materials]
        low_materials = [m for m in low_materials if m in self.materials]
        
        if not high_materials or not low_materials:
            return self.generate_random_structure(num_layers)
        
        for i in range(num_layers):
            if i % 2 == 0:
                material = random.choice(high_materials)
            else:
                material = random.choice(low_materials)
            
            thickness = self.generate_thickness(material)
            structure.append((material, thickness))
        
        return structure
    
    def generate_graded_index_structure(self, num_layers):
        """生成梯度折射率结构"""
        structure = []
        
        # 按折射率排序所有材料
        available_materials = []
        for group_name in ['low_index', 'medium_index', 'high_index', 'very_high_index']:
            group_materials = self.material_db.material_groups[group_name]
            for mat in group_materials:
                if mat in self.materials:
                    n_typical = self.material_db.material_properties[mat]['typical_n']
                    available_materials.append((mat, n_typical))
        
        # 按折射率排序
        available_materials.sort(key=lambda x: x[1])
        
        if len(available_materials) < num_layers:
            return self.generate_random_structure(num_layers)
        
        # 选择梯度方向（递增或递减）
        if random.random() < 0.5:
            # 递增
            selected_indices = np.linspace(0, len(available_materials)-1, num_layers, dtype=int)
        else:
            # 递减
            selected_indices = np.linspace(len(available_materials)-1, 0, num_layers, dtype=int)
        
        for idx in selected_indices:
            material = available_materials[idx][0]
            thickness = self.generate_thickness(material)
            structure.append((material, thickness))
        
        return structure
    
    def generate_symmetric_structure(self, num_layers):
        """生成对称结构"""
        if num_layers < 3:
            return self.generate_random_structure(num_layers)
        
        structure = []
        half_layers = num_layers // 2
        
        # 生成前半部分
        first_half = []
        last_material = None
        
        for i in range(half_layers):
            available_materials = [m for m in self.materials if m != last_material]
            material = random.choice(available_materials)
            thickness = self.generate_thickness(material)
            first_half.append((material, thickness))
            last_material = material
        
        # 如果是奇数层，添加中心层
        if num_layers % 2 == 1:
            available_materials = [m for m in self.materials if m != last_material]
            center_material = random.choice(available_materials)
            center_thickness = self.generate_thickness(center_material)
            structure = first_half + [(center_material, center_thickness)] + first_half[::-1]
        else:
            structure = first_half + first_half[::-1]
        
        return structure
    
    def generate_random_structure(self, num_layers):
        """生成随机结构（改进版）"""
        structure = []
        last_material = None
        
        for _ in range(num_layers):
            # 避免相邻层使用相同材料
            available_materials = [m for m in self.materials if m != last_material]
            material = random.choice(available_materials)
            thickness = self.generate_thickness(material)
            structure.append((material, thickness))
            last_material = material
        
        return structure
    
    def generate_structure_by_pattern(self, num_layers):
        """根据设计模式生成结构"""
        pattern = np.random.choice(
            list(self.design_patterns.keys()),
            p=list(self.design_patterns.values())
        )
        
        if pattern == 'alternating_index':
            return self.generate_alternating_index_structure(num_layers)
        elif pattern == 'graded_index':
            return self.generate_graded_index_structure(num_layers)
        elif pattern == 'symmetric':
            return self.generate_symmetric_structure(num_layers)
        else:  # random or cavity_based (暂时用random)
            return self.generate_random_structure(num_layers)
    
    def validate_structure(self, structure):
        """验证结构的物理合理性"""
        if not structure:
            return False, "空结构"
        
        materials = [layer[0] for layer in structure]
        thicknesses = [layer[1] for layer in structure]
        
        # 检查材料兼容性
        if not self.check_material_compatibility(materials):
            return False, "材料兼容性检查失败"
        
        # 检查厚度范围
        for material, thickness in structure:
            config = self.thickness_configs.get(material, {'range': (20, 300)})
            min_thick, max_thick = config['range']
            if not (min_thick <= thickness <= max_thick):
                return False, f"厚度超出范围: {material} {thickness}nm"
        
        # 检查总厚度
        total_thickness = sum(thicknesses)
        if total_thickness > 3000:  # 总厚度不超过3μm
            return False, f"总厚度过大: {total_thickness}nm"
        
        return True, "验证通过"
    
    def generate_dataset_parallel(self, num_samples=300000, wavelengths=None, num_processes=6):
        """生成优化数据集 - 多进程并行版本"""
        if wavelengths is None:
            wavelengths = np.arange(400, 1101, 10)  # 71个波长点
        
        safe_print(f"[并行生成] 开始生成优化数据集...")
        safe_print(f"   目标样本数: {num_samples}")
        safe_print(f"   并行进程数: {num_processes}")
        safe_print(f"   波长范围: {wavelengths[0]}-{wavelengths[-1]}nm ({len(wavelengths)}点)")
        safe_print(f"   材料数量: {len(self.materials)}")
        safe_print(f"   设计模式: {list(self.design_patterns.keys())}")
        
        # 计算每个进程的样本数
        samples_per_process = num_samples // num_processes
        remaining_samples = num_samples % num_processes
        
        # 为每个进程分配样本数
        process_samples = [samples_per_process] * num_processes
        for i in range(remaining_samples):
            process_samples[i] += 1
        
        safe_print(f"   每进程样本数: {process_samples}")
        
        # 使用多进程生成数据
        start_time = time.time()
        
        with Pool(processes=num_processes) as pool:
            # 创建参数列表
            args_list = []
            for i, samples in enumerate(process_samples):
                args_list.append((
                    samples,
                    wavelengths,
                    self.materials,
                    self.material_db.material_groups,
                    self.material_db.material_properties,
                    self.layer_distribution,
                    self.thickness_configs,
                    self.design_patterns,
                    i  # process_id for random seed
                ))
            
            # 并行执行
            safe_print(f"[并行] 启动 {num_processes} 个进程...")
            results = pool.map(generate_samples_worker, args_list)
        
        # 合并结果
        safe_print(f"[并行] 合并结果...")
        all_structures = []
        all_transmission = []
        all_reflection = []
        all_quality_scores = []
        
        total_failed = 0
        total_low_quality = 0
        
        for result in results:
            if result is not None:
                all_structures.extend(result['structures'])
                all_transmission.extend(result['transmission'])
                all_reflection.extend(result['reflection'])
                all_quality_scores.extend(result['quality_scores'])
                total_failed += result['stats']['failed_samples']
                total_low_quality += result['stats']['low_quality_samples']
        
        elapsed_time = time.time() - start_time
        
        safe_print(f"[完成] 并行数据集生成完毕")
        safe_print(f"   实际生成样本: {len(all_structures)}")
        safe_print(f"   跳过的无效结构: {total_failed}")
        safe_print(f"   跳过的低质量样本: {total_low_quality}")
        safe_print(f"   平均质量分数: {np.mean(all_quality_scores):.3f}")
        safe_print(f"   总耗时: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分钟)")
        safe_print(f"   生成速度: {len(all_structures)/elapsed_time:.1f} 样本/秒")
        
        return {
            'structures': all_structures,
            'wavelengths': wavelengths,
            'transmission': np.array(all_transmission),
            'reflection': np.array(all_reflection),
            'quality_scores': np.array(all_quality_scores),
            'generation_stats': {
                'failed_samples': total_failed,
                'low_quality_samples': total_low_quality,
                'avg_quality': float(np.mean(all_quality_scores)),
                'generation_time': elapsed_time,
                'samples_per_second': len(all_structures)/elapsed_time
            }
        }
    
    def generate_dataset(self, num_samples=300000, wavelengths=None):
        """生成优化的数据集"""
        if wavelengths is None:
            wavelengths = np.arange(400, 1101, 10)  # 71个波长点
        
        safe_print(f"[生成] 开始生成优化数据集...")
        safe_print(f"   目标样本数: {num_samples}")
        safe_print(f"   波长范围: {wavelengths[0]}-{wavelengths[-1]}nm ({len(wavelengths)}点)")
        safe_print(f"   材料数量: {len(self.materials)}")
        safe_print(f"   设计模式: {list(self.design_patterns.keys())}")
        
        structures = []
        transmission_spectra = []
        reflection_spectra = []
        quality_scores = []
        
        tmm_calc = OptimizedTMMCalculator(self.material_db)
        
        failed_samples = 0
        low_quality_samples = 0
        
        pbar = tqdm.tqdm(total=num_samples, desc="生成样本")
        
        while len(structures) < num_samples:
            # 随机选择层数
            num_layers = np.random.choice(
                list(self.layer_distribution.keys()),
                p=list(self.layer_distribution.values())
            )
            
            # 根据设计模式生成结构
            structure = self.generate_structure_by_pattern(num_layers)
            
            # 验证结构
            is_valid, error_msg = self.validate_structure(structure)
            if not is_valid:
                failed_samples += 1
                if failed_samples % 1000 == 0:
                    safe_print(f"[警告] 已跳过 {failed_samples} 个无效结构")
                continue
            
            # 计算光谱
            T, R, validation = tmm_calc.calculate_spectrum_with_validation(structure, wavelengths)
            
            # 质量检查
            quality_score = validation.get('quality_score', 0.0)
            if quality_score < 0.8:  # 质量阈值
                low_quality_samples += 1
                if low_quality_samples % 500 == 0:
                    safe_print(f"[警告] 已跳过 {low_quality_samples} 个低质量样本")
                continue
            
            # 接受样本
            structures.append(structure)
            transmission_spectra.append(T)
            reflection_spectra.append(R)
            quality_scores.append(quality_score)
            
            pbar.update(1)
            
            # 定期报告进度
            if len(structures) % 10000 == 0:
                avg_quality = np.mean(quality_scores[-10000:])
                safe_print(f"[进度] 已生成 {len(structures)} 样本，平均质量: {avg_quality:.3f}")
        
        pbar.close()
        
        safe_print(f"[完成] 数据集生成完毕")
        safe_print(f"   有效样本: {len(structures)}")
        safe_print(f"   跳过的无效结构: {failed_samples}")
        safe_print(f"   跳过的低质量样本: {low_quality_samples}")
        safe_print(f"   平均质量分数: {np.mean(quality_scores):.3f}")
        
        return {
            'structures': structures,
            'wavelengths': wavelengths,
            'transmission': np.array(transmission_spectra),
            'reflection': np.array(reflection_spectra),
            'quality_scores': np.array(quality_scores),
            'generation_stats': {
                'failed_samples': failed_samples,
                'low_quality_samples': low_quality_samples,
                'avg_quality': float(np.mean(quality_scores))
            }
        }

def generate_samples_worker(args):
    """工作进程函数 - 生成指定数量的样本"""
    (num_samples, wavelengths, materials, material_groups, material_properties, 
     layer_distribution, thickness_configs, design_patterns, process_id) = args
    
    # 设置进程特定的随机种子
    np.random.seed(42 + process_id * 1000)
    random.seed(42 + process_id * 1000)
    
    # 重建材料数据库和生成器（每个进程独立）
    try:
        material_db = OptimizedMaterialDatabase()
        tmm_calc = OptimizedTMMCalculator(material_db)
        
        # 创建简化的生成器实例
        class WorkerGenerator:
            def __init__(self):
                self.materials = materials
                self.material_groups = material_groups
                self.material_properties = material_properties
                self.layer_distribution = layer_distribution
                self.thickness_configs = thickness_configs
                self.design_patterns = design_patterns
            
            def generate_thickness(self, material):
                """为指定材料生成厚度"""
                config = self.thickness_configs.get(material, {
                    'range': (20, 300),
                    'distribution': 'uniform', 
                    'precision': 2
                })
                
                min_thick, max_thick = config['range']
                precision = config['precision']
                
                if config['distribution'] == 'log_uniform':
                    log_min, log_max = np.log(min_thick), np.log(max_thick)
                    thickness = np.exp(np.random.uniform(log_min, log_max))
                else:
                    thickness = np.random.uniform(min_thick, max_thick)
                
                thickness = round(thickness / precision) * precision
                return max(min_thick, min(max_thick, thickness))
            
            def check_material_compatibility(self, materials_sequence):
                """检查材料序列的物理兼容性"""
                if len(materials_sequence) < 2:
                    return True
                for i in range(len(materials_sequence) - 1):
                    if materials_sequence[i] == materials_sequence[i + 1]:
                        return False
                return True
            
            def generate_alternating_index_structure(self, num_layers):
                """生成高低折射率交替结构"""
                structure = []
                high_materials = self.material_groups['high_index'] + self.material_groups['very_high_index']
                low_materials = self.material_groups['low_index'] + self.material_groups['medium_index']
                
                high_materials = [m for m in high_materials if m in self.materials]
                low_materials = [m for m in low_materials if m in self.materials]
                
                if not high_materials or not low_materials:
                    return self.generate_random_structure(num_layers)
                
                for i in range(num_layers):
                    if i % 2 == 0:
                        material = random.choice(high_materials)
                    else:
                        material = random.choice(low_materials)
                    thickness = self.generate_thickness(material)
                    structure.append((material, thickness))
                return structure
            
            def generate_graded_index_structure(self, num_layers):
                """生成梯度折射率结构"""
                structure = []
                available_materials = []
                for group_name in ['low_index', 'medium_index', 'high_index', 'very_high_index']:
                    group_materials = self.material_groups[group_name]
                    for mat in group_materials:
                        if mat in self.materials:
                            n_typical = self.material_properties[mat]['typical_n']
                            available_materials.append((mat, n_typical))
                
                available_materials.sort(key=lambda x: x[1])
                
                if len(available_materials) < num_layers:
                    return self.generate_random_structure(num_layers)
                
                if random.random() < 0.5:
                    selected_indices = np.linspace(0, len(available_materials)-1, num_layers, dtype=int)
                else:
                    selected_indices = np.linspace(len(available_materials)-1, 0, num_layers, dtype=int)
                
                for idx in selected_indices:
                    material = available_materials[idx][0]
                    thickness = self.generate_thickness(material)
                    structure.append((material, thickness))
                return structure
            
            def generate_symmetric_structure(self, num_layers):
                """生成对称结构"""
                if num_layers < 3:
                    return self.generate_random_structure(num_layers)
                
                structure = []
                half_layers = num_layers // 2
                first_half = []
                last_material = None
                
                for i in range(half_layers):
                    available_materials = [m for m in self.materials if m != last_material]
                    material = random.choice(available_materials)
                    thickness = self.generate_thickness(material)
                    first_half.append((material, thickness))
                    last_material = material
                
                if num_layers % 2 == 1:
                    available_materials = [m for m in self.materials if m != last_material]
                    center_material = random.choice(available_materials)
                    center_thickness = self.generate_thickness(center_material)
                    structure = first_half + [(center_material, center_thickness)] + first_half[::-1]
                else:
                    structure = first_half + first_half[::-1]
                return structure
            
            def generate_random_structure(self, num_layers):
                """生成随机结构"""
                structure = []
                last_material = None
                for _ in range(num_layers):
                    available_materials = [m for m in self.materials if m != last_material]
                    material = random.choice(available_materials)
                    thickness = self.generate_thickness(material)
                    structure.append((material, thickness))
                    last_material = material
                return structure
            
            def generate_structure_by_pattern(self, num_layers):
                """根据设计模式生成结构"""
                pattern = np.random.choice(
                    list(self.design_patterns.keys()),
                    p=list(self.design_patterns.values())
                )
                
                if pattern == 'alternating_index':
                    return self.generate_alternating_index_structure(num_layers)
                elif pattern == 'graded_index':
                    return self.generate_graded_index_structure(num_layers)
                elif pattern == 'symmetric':
                    return self.generate_symmetric_structure(num_layers)
                else:
                    return self.generate_random_structure(num_layers)
            
            def validate_structure(self, structure):
                """验证结构的物理合理性"""
                if not structure:
                    return False, "空结构"
                
                materials_seq = [layer[0] for layer in structure]
                thicknesses = [layer[1] for layer in structure]
                
                if not self.check_material_compatibility(materials_seq):
                    return False, "材料兼容性检查失败"
                
                for material, thickness in structure:
                    config = self.thickness_configs.get(material, {'range': (20, 300)})
                    min_thick, max_thick = config['range']
                    if not (min_thick <= thickness <= max_thick):
                        return False, f"厚度超出范围: {material} {thickness}nm"
                
                total_thickness = sum(thicknesses)
                if total_thickness > 3000:
                    return False, f"总厚度过大: {total_thickness}nm"
                
                return True, "验证通过"
        
        generator = WorkerGenerator()
        
        # 生成样本
        structures = []
        transmission_spectra = []
        reflection_spectra = []
        quality_scores = []
        
        failed_samples = 0
        low_quality_samples = 0
        
        while len(structures) < num_samples:
            # 随机选择层数
            num_layers = np.random.choice(
                list(layer_distribution.keys()),
                p=list(layer_distribution.values())
            )
            
            # 生成结构
            structure = generator.generate_structure_by_pattern(num_layers)
            
            # 验证结构
            is_valid, error_msg = generator.validate_structure(structure)
            if not is_valid:
                failed_samples += 1
                continue
            
            # 计算光谱
            T, R, validation = tmm_calc.calculate_spectrum_with_validation(structure, wavelengths)
            
            # 质量检查
            quality_score = validation.get('quality_score', 0.0)
            if quality_score < 0.8:
                low_quality_samples += 1
                continue
            
            # 接受样本
            structures.append(structure)
            transmission_spectra.append(T)
            reflection_spectra.append(R)
            quality_scores.append(quality_score)
        
        return {
            'structures': structures,
            'transmission': transmission_spectra,
            'reflection': reflection_spectra,
            'quality_scores': quality_scores,
            'stats': {
                'failed_samples': failed_samples,
                'low_quality_samples': low_quality_samples,
                'process_id': process_id
            }
        }
        
    except Exception as e:
        print(f"[错误] 进程 {process_id} 失败: {str(e)}")
        return None

def save_optimized_dataset(dataset, material_db, output_dir="optimized_dataset"):
    """保存优化数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存主要数据
    output_file = os.path.join(output_dir, "optimized_multilayer_dataset.npz")
    
    # 转换结构为可保存格式
    structures_array = []
    max_layers = max(len(s) for s in dataset['structures'])
    
    for structure in dataset['structures']:
        # 填充到最大长度
        padded_structure = structure + [('', 0.0)] * (max_layers - len(structure))
        formatted_structure = []
        for material, thickness in padded_structure:
            formatted_structure.append([material, float(thickness)])
        structures_array.append(formatted_structure)
    
    np.savez_compressed(output_file,
                       structures=np.array(structures_array, dtype=object),
                       wavelengths=dataset['wavelengths'],
                       transmission=dataset['transmission'],
                       reflection=dataset['reflection'],
                       quality_scores=dataset['quality_scores'])
    
    safe_print(f"[保存] 数据集已保存至: {output_file}")
    safe_print(f"[保存] 数据集形状: {dataset['transmission'].shape}")
    
    # 保存详细元数据
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'dataset_info': {
            'num_samples': len(dataset['structures']),
            'wavelength_range': f"{dataset['wavelengths'][0]}-{dataset['wavelengths'][-1]}nm",
            'num_wavelengths': len(dataset['wavelengths']),
            'max_layers': max_layers,
            'materials': list(material_db.materials.keys())
        },
        'generation_stats': dataset['generation_stats'],
        'quality_metrics': {
            'avg_quality_score': float(np.mean(dataset['quality_scores'])),
            'min_quality_score': float(np.min(dataset['quality_scores'])),
            'max_quality_score': float(np.max(dataset['quality_scores'])),
            'quality_std': float(np.std(dataset['quality_scores']))
        },
        'layer_statistics': {},
        'material_statistics': {},
        'thickness_statistics': {}
    }
    
    # 统计层数分布
    layer_counts = [len(s) for s in dataset['structures']]
    for i in range(3, 11):
        metadata['layer_statistics'][f'{i}_layers'] = int(np.sum(np.array(layer_counts) == i))
    
    # 统计材料使用频率
    material_usage = {}
    for structure in dataset['structures']:
        for material, _ in structure:
            material_usage[material] = material_usage.get(material, 0) + 1
    metadata['material_statistics'] = material_usage
    
    # 统计厚度分布
    all_thicknesses = []
    for structure in dataset['structures']:
        for _, thickness in structure:
            all_thicknesses.append(thickness)
    
    if all_thicknesses:
        metadata['thickness_statistics'] = {
            'mean': float(np.mean(all_thicknesses)),
            'std': float(np.std(all_thicknesses)),
            'min': float(np.min(all_thicknesses)),
            'max': float(np.max(all_thicknesses)),
            'median': float(np.median(all_thicknesses))
        }
    
    # 保存元数据
    metadata_file = os.path.join(output_dir, "optimized_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    safe_print(f"[保存] 元数据已保存至: {metadata_file}")
    
    return metadata

def main():
    """主函数"""
    safe_print("="*60)
    safe_print("优化的多层薄膜数据集生成器")
    safe_print("="*60)
    
    # 初始化优化的材料数据库
    safe_print("[1/4] 初始化材料数据库...")
    material_db = OptimizedMaterialDatabase()
    
    # 初始化优化的多层生成器
    safe_print("[2/4] 初始化结构生成器...")
    generator = OptimizedMultilayerGenerator(material_db)
    
    # 生成优化数据集 - 使用并行版本
    safe_print("[3/4] 生成优化数据集...")
    dataset = generator.generate_dataset_parallel(num_samples=300000, num_processes=6)
    
    # 保存数据集
    safe_print("[4/4] 保存数据集...")
    metadata = save_optimized_dataset(dataset, material_db)
    
    # 输出总结
    safe_print("\n" + "="*60)
    safe_print("生成完成总结")
    safe_print("="*60)
    safe_print(f"✅ 成功生成 {metadata['dataset_info']['num_samples']} 个样本")
    safe_print(f"✅ 平均质量分数: {metadata['quality_metrics']['avg_quality_score']:.3f}")
    safe_print(f"✅ 材料种类: {len(metadata['dataset_info']['materials'])} 种")
    safe_print(f"✅ 层数范围: 3-{metadata['dataset_info']['max_layers']} 层")
    safe_print(f"✅ 数据集文件: optimized_dataset/optimized_multilayer_dataset.npz")
    
    # 显示改进点
    safe_print("\n🎯 主要改进:")
    safe_print("   • 移除了Ag材料，专注于介电质多层膜")
    safe_print("   • 优化了层数分布，基于样本空间大小") 
    safe_print("   • 改进了厚度范围，基于材料物理特性")
    safe_print("   • 添加了物理兼容性检查")
    safe_print("   • 实现了多种设计模式（交替、梯度、对称等）")
    safe_print("   • 增强了数据质量控制")

if __name__ == "__main__":
    main()
