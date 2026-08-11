/* X7 TensorRT path: sparse InputPlane expand on GPU.
 *
 * Layout follows px0/lc0 onnx_kernels.cu (xiangqi: 128-bit mask + 90 squares).
 * Device code uses two uint64 words instead of MSVC host-only std::_Unsigned128.
 */

#include "onnx_kernels.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned kSquares = 90; /* 10x9 */

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct Mask128 {
  uint64_t lo;
  uint64_t hi;
};

__device__ __forceinline__ Mask128 mask_shr(Mask128 mask, unsigned shift) {
  if (shift == 0) {
    return mask;
  }
  if (shift >= 128) {
    return Mask128{0, 0};
  }
  if (shift >= 64) {
    return Mask128{mask.hi >> (shift - 64), 0};
  }
  return Mask128{(mask.lo >> shift) | (mask.hi << (64 - shift)), mask.hi >> shift};
}

__device__ __forceinline__ bool mask_bit0(Mask128 mask) { return (mask.lo & 1ull) != 0ull; }

template <unsigned bits_per_thread>
__global__ void expand_planes_kernel(float* output, const Mask128* masks, const float* values,
                                     unsigned n) {
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
  index *= bits_per_thread;
  unsigned plane_index = index / kSquares;
  if (plane_index >= n) {
    return;
  }

  Mask128 mask = masks[plane_index];
  unsigned sq_index = index % kSquares;
  float value = values[plane_index];
  float op[bits_per_thread] = {};
  mask = mask_shr(mask, sq_index);
  for (unsigned i = 0; i < bits_per_thread; ++i) {
    if (mask_bit0(mask)) {
      op[i] = value;
    }
    mask = mask_shr(mask, 1);
  }
  for (unsigned i = 0; i < bits_per_thread; ++i) {
    output[index + i] = op[i];
  }
}

}  // namespace

extern "C" int x7_cuda_stream_create(void** stream_out) {
  cudaStream_t stream = nullptr;
  cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *stream_out = stream;
  return 0;
}

extern "C" int x7_cuda_stream_destroy(void* stream) {
  if (stream == nullptr) {
    return 0;
  }
  return static_cast<int>(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}

extern "C" int x7_cuda_stream_synchronize(void* stream) {
  return static_cast<int>(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

extern "C" int x7_cuda_malloc(void** ptr_out, size_t bytes) {
  return static_cast<int>(cudaMalloc(ptr_out, bytes));
}

extern "C" int x7_cuda_free(void* ptr) {
  if (ptr == nullptr) {
    return 0;
  }
  return static_cast<int>(cudaFree(ptr));
}

extern "C" int x7_cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
  return static_cast<int>(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(stream)));
}

extern "C" int x7_expand_planes_f32(float* output, const void* packed, unsigned n,
                                    void* stream) {
  constexpr unsigned bits_per_thread = 2;
  const int threads = static_cast<int>(n * kSquares / bits_per_thread);
  constexpr int block_size = 256;
  const int blocks = DivUp(threads, block_size);

  const auto* masks = static_cast<const Mask128*>(packed);
  const float* values = reinterpret_cast<const float*>(masks + n);

  expand_planes_kernel<bits_per_thread>
      <<<blocks, block_size, 0, static_cast<cudaStream_t>(stream)>>>(output, masks, values, n);
  return static_cast<int>(cudaGetLastError());
}
