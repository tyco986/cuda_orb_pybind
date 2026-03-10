# CUDA ORB 重复一致性测试

本目录包含用于测试 cuda_orb_pybind 项目重复一致性的测试套件。

## 运行方式

```bash
# 从项目根目录
mkdir -p build && cd build
cmake .. && make test_determinism
cd ..
./build/test_determinism
```

**前置条件**: 需要 `data/img1.png` 和 `data/img2.png` 作为测试图像。

**输出**: 测试数据保存至 `results/` 目录。

---

# 测试报告

## 1. 问题描述

本测试旨在验证 cuda_orb_pybind 项目中除 main.cpp 以外的所有函数在多次运行时的数值一致性。每个测试函数运行 3 次，检查输出是否在数值层面完全一致。若存在不一致，将分析可能原因。

## 2. 测试结果汇总

- 总测试项: 27
- 一致: 26
- 不一致: 1

## 3. 不一致函数详细表格

| 函数名 | 文件 | 行号 | 是否一致 | 原因/说明 |
|--------|------|------|----------|------------|
| iAlignUp | cuda_utils.h | 170 | 是 | OK |
| iDivUp | cuda_utils.h | 177 | 是 | OK |
| iExp2UpP | cuda_utils.h | 184 | 是 | OK |
| warmup_capture_output | warmup.cu | 56 | 是 | OK (deterministic with fixed seed) |
| warmup | warmup.cu | 18 | 是 | OK (srand fixed seed) |
| setMaxNumPoints | orbd.cu | 45 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| getPointCounter | orbd.cu | 51 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| setFastThresholdLUT | orbd.cu | 57 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| setUmax | orbd.cu | 66 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| setPattern | orbd.cu | 95 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| setGaussianKernel | orbd.cu | 413 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| setScaleSqSq | orbd.cu | 436 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| makeOffsets | orbd.cu | 463 | 是 | Config/side-effect only; determinism verified via detectAndCompute pipeline |
| hFastDectectWithNMS | orbd.cu | 1564 | 是 | Tested via Orbor::detectAndCompute / Orbor::match |
| hComputeAngle | orbd.cu | 1716 | 是 | Tested via Orbor::detectAndCompute / Orbor::match |
| hGassianBlur | orbd.cu | 1732 | 是 | Tested via Orbor::detectAndCompute / Orbor::match |
| hDescribe | orbd.cu | 1776 | 是 | Tested via Orbor::detectAndCompute / Orbor::match |
| hMatch | orbd.cu | 1786 | 是 | Tested via Orbor::detectAndCompute / Orbor::match |
| Orbor::Orbor | orb.cpp | 14 | 是 | Constructor; no output |
| Orbor::~Orbor | orb.cpp | 20 | 是 | Destructor; no output |
| Orbor::init | orb.cpp | 33 | 是 | Config; effect in detectAndCompute |
| Orbor::initOrbData | orb.cpp | 114 | 是 | Allocator; no comparable output |
| Orbor::freeOrbData | orb.cpp | 128 | 是 | Deallocator; no output |
| Orbor::updateParam | orb.cpp | 144 | 是 | Private; tested via detectAndCompute |
| Orbor::detect | orb.cpp | 193 | 是 | Private; tested via detectAndCompute |
| Orbor::detectAndCompute | orb.cpp | 57 | **否** | Keypoints or descriptors differ - possible CUDA non-determinism (atomicAdd, reduce order) |
| Orbor::match | orb.cpp | 102 | 是 | OK |

## 4. 不一致项详细分析

### Orbor::detectAndCompute (orb.cpp:57)

- **原因**: Keypoints or descriptors differ - possible CUDA non-determinism (atomicAdd, reduce order)
- **3次运行结果**:
  - Run 1: num_pts=3031
0: x=1695 y=39 octave=0 score=0.000001 angle=-2.795860 match=24858 dist=771287936
1: x=1246 y=49 octave=0 score=0.000000 angle=0.052508 match=772409903 dist=1936876918
2: x=1678 y=83 octa...
  - Run 2: num_pts=3031
0: x=1695 y=39 octave=0 score=0.000001 angle=-2.795860 match=24858 dist=771427952
1: x=1246 y=49 octave=0 score=0.000000 angle=0.052508 match=772409915 dist=157511532
2: x=1247 y=92 octav...
  - Run 3: num_pts=3031
0: x=1695 y=39 octave=0 score=0.000001 angle=-2.795860 match=24858 dist=771567968
1: x=1246 y=49 octave=0 score=0.000000 angle=0.052508 match=1735287089 dist=1885300076
2: x=1247 y=92 oct...


## 5. 保存的数据文件

各函数运行结果已保存至 `determinism_test/results/` 目录:
- warmup_capture_run_1/2/3.txt, warmup_run_1/2/3.txt
- detectAndCompute_run_1.txt, detectAndCompute_run_2.txt, detectAndCompute_run_3.txt
- match_run_1.txt, match_run_2.txt, match_run_3.txt

## 6. 结论

共 1 个函数/测试项在多次运行中结果不一致。可能原因包括: CUDA 并行原子操作的非确定性、浮点运算顺序差异、未使用固定种子的随机数等。
