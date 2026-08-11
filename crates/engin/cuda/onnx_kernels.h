#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 0 = success; non-zero = cudaError_t. */
int x7_cuda_stream_create(void** stream_out);
int x7_cuda_stream_destroy(void* stream);
int x7_cuda_stream_synchronize(void* stream);

int x7_cuda_malloc(void** ptr_out, size_t bytes);
int x7_cuda_free(void* ptr);

int x7_cuda_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
int x7_cuda_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);

/* packed = [n * u128 masks][n * f32 values]; output = n * 90 floats (NCHW plane layout). */
int x7_expand_planes_f32(float* output, const void* packed, unsigned n, void* stream);

#ifdef __cplusplus
}
#endif
