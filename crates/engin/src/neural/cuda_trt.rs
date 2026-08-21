//! TensorRT 用 CUDA：stream、device/pinned buffer、sparse→dense expand。
//! Kernel 对齐 px0 `onnx_kernels.cu`（128-bit mask + 90 格）。

use std::ffi::c_void;
use std::ptr;
use std::slice;

use crate::EnginError;

unsafe extern "C" {
    fn x7_cuda_stream_create(out: *mut *mut c_void) -> i32;
    fn x7_cuda_stream_destroy(stream: *mut c_void) -> i32;
    fn x7_cuda_stream_synchronize(stream: *mut c_void) -> i32;
    fn x7_cuda_malloc(out: *mut *mut c_void, bytes: usize) -> i32;
    fn x7_cuda_free(ptr: *mut c_void) -> i32;
    fn x7_cuda_host_alloc(out: *mut *mut c_void, bytes: usize, flags: i32) -> i32;
    fn x7_cuda_host_free(ptr: *mut c_void) -> i32;
    fn x7_cuda_memcpy_h2d_async(dst: *mut c_void, src: *const c_void, bytes: usize, stream: *mut c_void) -> i32;
    fn x7_cuda_memcpy_d2h_async(dst: *mut c_void, src: *const c_void, bytes: usize, stream: *mut c_void) -> i32;
    fn x7_expand_planes_f32(output: *mut f32, packed: *const c_void, n: u32, stream: *mut c_void) -> i32;
}

fn check(status: i32, what: &str) -> Result<(), EnginError> {
    (status == 0)
        .then_some(())
        .ok_or_else(|| EnginError::Onnx(format!("CUDA {what}: {status}")))
}

pub struct CudaStream(*mut c_void);

impl CudaStream {
    pub fn new() -> Result<Self, EnginError> {
        let mut ptr = ptr::null_mut();
        check(unsafe { x7_cuda_stream_create(&mut ptr) }, "stream_create")?;
        Ok(Self(ptr))
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }

    pub fn sync(&self) -> Result<(), EnginError> {
        check(unsafe { x7_cuda_stream_synchronize(self.0) }, "sync")
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = unsafe { x7_cuda_stream_destroy(self.0) };
    }
}

unsafe impl Send for CudaStream {}

pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self, EnginError> {
        let mut ptr = ptr::null_mut();
        check(unsafe { x7_cuda_malloc(&mut ptr, bytes) }, "malloc")?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn upload_async(&self, src: &[u8], stream: &CudaStream) -> Result<(), EnginError> {
        if src.len() > self.bytes {
            return Err(EnginError::Onnx("CUDA H2D overflow".into()));
        }
        check(
            unsafe { x7_cuda_memcpy_h2d_async(self.ptr, src.as_ptr().cast(), src.len(), stream.as_ptr()) },
            "H2D",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = unsafe { x7_cuda_free(self.ptr) };
    }
}

unsafe impl Send for DeviceBuffer {}

/// `cudaHostAlloc` 钉扎缓冲；`write_combined` 仅适合 CPU 写、H2D 读。
pub struct PinnedBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl PinnedBuffer {
    pub fn new(bytes: usize, write_combined: bool) -> Result<Self, EnginError> {
        let mut ptr = ptr::null_mut();
        let flags = if write_combined { 1 } else { 0 };
        check(unsafe { x7_cuda_host_alloc(&mut ptr, bytes, flags) }, "host_alloc")?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.cast(), self.bytes) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.cast(), self.bytes) }
    }

    pub fn as_f32(&self) -> &[f32] {
        let n = self.bytes / size_of::<f32>();
        unsafe { slice::from_raw_parts(self.ptr.cast(), n) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let _ = unsafe { x7_cuda_host_free(self.ptr) };
    }
}

unsafe impl Send for PinnedBuffer {}

pub fn expand_planes_async(
    dense: &DeviceBuffer,
    sparse: &DeviceBuffer,
    n_planes: u32,
    stream: &CudaStream,
) -> Result<(), EnginError> {
    check(
        unsafe { x7_expand_planes_f32(dense.ptr.cast(), sparse.ptr, n_planes, stream.as_ptr()) },
        "expand",
    )
}

/// 从任意 device 指针异步 D2H（ORT 输出 Tensor 的 data_ptr）。
pub fn download_device_async(dst: &mut [u8], src: *const c_void, stream: &CudaStream) -> Result<(), EnginError> {
    check(
        unsafe { x7_cuda_memcpy_d2h_async(dst.as_mut_ptr().cast(), src as *mut c_void, dst.len(), stream.as_ptr()) },
        "D2H",
    )
}
