# CUDA-ORB

GPU 加速的 ORB（Oriented FAST and Rotated BRIEF）特征检测、描述子计算与 Hamming 匹配，支持图像配准与单应性估计。

## 功能概述

- **ORB 特征**：FAST 角点检测、多尺度金字塔、Harris 响应排序、BRIEF 描述子
- **GPU 加速**：检测、描述、匹配均在 CUDA 上执行
- **图像配准**：ORB 匹配 + RANSAC 单应性估计，输出 3×3 变换矩阵
- **Python 与 C++**：提供 `cuda_orb` Python 包和 `cuda_orb` 可执行文件

## 项目结构

```
cuda_orb_pybind/
├── src/                    # C++/CUDA 源码
│   ├── orbd.cu, orbd.h     # ORB CUDA 内核
│   ├── orb.cpp, orb.h      # Orbor 类
│   ├── orb_bindings.cpp    # pybind11 绑定
│   └── ...
├── cuda_orb_cpp/           # C++ 应用
│   ├── orb_aligner.cpp     # OrbAligner C++ 实现
│   ├── orb_aligner.h
│   └── test_orb_aligner.cpp   # C++ benchmark
├── cuda_orb/               # Python 包
│   ├── __init__.py
│   ├── orb_aligner.py      # OrbAligner 高层接口
│   ├── test_orb_aligner.py # Python benchmark
│   └── _cuda_orb.so        # 编译生成的扩展
├── example_data/
│   ├── image.png
│   └── template.png
├── CMakeLists.txt
├── pyproject.toml
└── setup.py
```

## 依赖

- **CUDA**（含 nvcc）
- **OpenCV**（≥ 4.x）
- **CMake** ≥ 3.18
- **Python** ≥ 3.8（如需 Python 包）
- **pybind11**（如需 Python 包）

## 编译

### 1. C++ 可执行文件 `cuda_orb`

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make cuda_orb -j$(nproc)
```

生成的可执行文件：`build/cuda_orb`

### 2. Python 包 `cuda_orb`

```bash
pip install -e . --no-build-isolation
```

或使用 `setup.py`：

```bash
python setup.py develop
```

## 运行

### C++ 可执行文件

```bash
./build/cuda_orb [选项]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--image PATH` | 输入图像路径 | `example_data/image.png` |
| `--template PATH` | 模板图像路径 | `example_data/template.png` |
| `--batch N` | 批大小 | 8 |
| `--device N` | CUDA 设备 ID | 0 |
| `--no-nndr` | 关闭 NNDR 过滤 | - |

**示例：**

```bash
# 使用默认图像，batch=8
./build/cuda_orb

# 指定图像与 batch
./build/cuda_orb --image example_data/image.png --template example_data/template.png --batch 4

# 指定 GPU
./build/cuda_orb --device 1 --batch 8
```

**输出：** GPU/CPU/内存峰值、总耗时、每样本耗时、各样本的 3×3 warp 矩阵。

### Python 包

```python
import cv2
import cuda_orb

# 创建配准器
aligner = cuda_orb.OrbAligner(device=0)

# 单对图像
template = cv2.imread("example_data/template.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("example_data/image.png", cv2.IMREAD_GRAYSCALE)
result = aligner.find_transform(template[np.newaxis], image[np.newaxis])
H = result["warp_matrix"][0]  # 3×3 单应性矩阵
```

**批量 benchmark：**

```bash
python cuda_orb/test_orb_aligner.py --image example_data/image.png --template example_data/template.png --batch 8
```

## 参考

- [OpenCV ORB](https://github.com/opencv/opencv)
- [ORB: An efficient alternative to SIFT or SURF](http://www.gwylab.com/download/ORB_2012.pdf)
