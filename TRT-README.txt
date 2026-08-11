X7 TensorRT bundle (Windows x64)
================================

前置条件
--------

- 仅支持 Windows x64 的 NVIDIA 显卡；AMD、Intel 或没有 NVIDIA GPU 的机器请使用 DirectML 包。
- GPU 至少需要 Turing 架构（计算能力 SM 7.5，例如 GTX 16 / RTX 20 系列或更新）。
- 先更新 NVIDIA 显卡驱动。CUDA 13 建议 r580 或更新版本；安装后运行 `nvidia-smi`，
  能正确显示显卡与驱动版本才继续。
- 驱动下载：https://www.nvidia.com/Download/index.aspx

运行前安装(如果没有)
----------

1. CUDA Toolkit 13.x（本包以 CUDA 13.3 开发和验证）
   - 下载：https://developer.nvidia.com/cuda-downloads

2. cuDNN 9（与 CUDA 13 对应）
   - 下载：https://developer.nvidia.com/cudnn-downloads

3. Microsoft Visual C++ x64 Runtime
   - 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe

将 CUDA 的 bin 和 cuDNN 的 bin 加入系统 PATH，然后重新打开图形界面或终端。例如：

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin
  C:\path\to\cudnn\bin

不需要设置 CUDA_PATH，也不需要把 DLL 手工复制到 CUDA 目录；保持安装目录并加入 PATH 即可。

首次运行
--------

首次加载模型会在本目录的 trt_cache 中为当前显卡构建 TensorRT engine，可能较慢。
请保持该目录可写。更换显卡、驱动、CUDA 或模型后，如有异常可删除 trt_cache 后重试；
不要从其他电脑复制该缓存。

排查
----

若提示缺少 cudart / cublas / cudnn DLL，检查 CUDA 13 与 cuDNN 9 是否为 x64，且上述 bin
目录已在 PATH。若提示 nvinfer DLL，确认仍使用完整本包，未删除其中 TensorRT 10 DLL。
