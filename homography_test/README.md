# Homography Test

比较本项目的 CUDA ORB + cv::findHomography 与 OpenCV ORB + cv::findHomography 的透视变换矩阵。

## 运行方式

```bash
# 从项目根目录
mkdir -p build && cd build
cmake .. && make homography_test
cd ..
./build/homography_test [img1] [img2]
```

**默认图像**: `data/img1.png`, `data/img2.png`

**输出**: 每次运行打印两种方法的透视变换矩阵 (3x3)。
