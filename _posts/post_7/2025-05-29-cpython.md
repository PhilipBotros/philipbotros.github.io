---
layout: post
title: "How to run CUDA (C) from Python"
date: 2025-04-20
cover: /images/post_6/cover.jpg
background: /images/post_5/bg.jpg
---

I've been quite interested in learning the details of the CUDA programming model lately. I'll write my findings in a later blog post, if you are interested in the repo check X.

After running these kernels using the standard NVCC compiler flow I was curious how frameworks like PyTorch use the Python C integration to xyz. I'll show a few ways to run CUDA kernels from Python.

### Calling CUDA from Python

As you'll probably know, Python provides ways to call C code to offload heavy computations to C libraries (think NumPy, PyTorch, etc). 

Given that most modern Python runs using cPython it'll give you a good idea on why this is possible.

Let's think about what actually happens when we are invoking a Python program:

1. the cPython binary is started
2. the interpreter reads and compiles the code to bytecode
3. the CPython virtual machine executes the bytecode line by line
4. each specific bytecode ops is translated to the corresponding C function
5. the precompiled C function runs on the machine

Hence we are by default already executing C, the leap to a C extension is not that hard to make.

Easy C integration is one of the reasons Python is such a popular language for scientific computing: write application logic in easy to write Python, offload the heavy lifting to C libraries.

There are two main ways Python provides to call C code:

- ctypes
- Python C API

For a simple external function call, you can use the `ctypes` library, which dynamically loads shared libraries at runtime.

More extensive integrations, such as defining new Python types in C or managing memory manually, are handled by the Python C API.

The main difference between the two is that ctypes works with pre-compiled shared libraries and does not interact with Python’s memory management, whereas the Python C API allows direct manipulation of Python objects, including reference counting and garbage collection. (REMOVE)

Talk about ABI?

#### ctypes

Let's start with the `ctypes` library.

Using this very simple sum kernel:
https://github.com/PhilipBotros/cudafun/blob/main/python/pykernel_sum.cu
``` C
extern "C" {
#include <cuda_runtime.h>

__global__ void add_arrays(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}

void launch_sum_kernel(const float* a, const float* b, float* out, int n) {
    float *d_a, *d_b, *d_out;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_arrays<<<blocks, threads>>>(d_a, d_b, d_out, n);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
}
```

We first need to compile the CUDA kernel into a shared object file, we can use Nvidia's compiler `nvcc` to do this:

```bash
nvcc -Xcompiler -fPIC -shared -o libsum.so pykernel_sum.cu
```

We can then use the `ctypes` library to load the shared object file and call the functions.
Example 1:

```python
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Load shared library created using:
# nvcc -Xcompiler -fPIC -shared -o libsum.so pykernel_sum.cu
lib = ctypes.CDLL('./libsum.so')

# Configure function signature
lib.launch_sum_kernel.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.launch_sum_kernel.restype = None

# Allocate arrays; NumPy arrays are wrappers around C arrays and are contiguous 
# and we can use them directly (provided we did not use any reshaping or indexing 
# that would change the memory layout to be non-contiguous)
N = 1024
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
out = np.zeros_like(a)

# Call the kernel and verify the result
lib.launch_sum_kernel(a, b, out, N)

np.testing.assert_allclose(out, a + b, rtol=1e-5)
print("Success, same result!")
```
What actually happens under the hood?

1. Python loads the CUDA shared object (libsum.so) using `dlopen()` and passes NumPy arrays as raw C pointers to the kernel, which executes in native C/CUDA.

2. CUDA launches the kernel on the GPU (add_arrays<<<grid, block>>>), where different threads process a[i] * b[i] in parallel, modifying the out array directly in memory.

Hence the beauty of heterogeneous programming, and, in this case, cross-language execution. 
Python seamlessly integrates with CUDA, allowing us to write high-level Python executing on the CPU, while the CUDA kernel is executed on the GPU.

#### PyTorch Extension
For the second method, we are going to use PyTorch's extension API.

We can re-use most of the initial kernel but can strip the memory management bit as PyTorch will take care of this.

```c
#include <cuda_runtime.h>

__global__ void add_arrays(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}

void launch_sum_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_arrays<<<blocks, threads, 0, stream>>>(a, b, out, n);
}
```
We'll need to write some glue code in C++ 

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_sum_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream);

torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "input sizes must match");

    auto out = torch::empty_like(a);
    int n = a.numel();

    launch_sum_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        c10::cuda::getCurrentCUDAStream()
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_cuda", &sum_cuda, "CUDA elementwise sum");
}
```
In Python, it's very straightforward to load the extension and run it.

```python
import torch
from torch.utils.cpp_extension import load

sum_ext = load(
    name="sum_cuda_ext",
    sources=["ptkernel_wrapper.cpp", "ptkernel_sum.cu"],
    verbose=True,
)

a = torch.randn(1024, device='cuda', dtype=torch.float32)
b = torch.randn(1024, device='cuda', dtype=torch.float32)
out = sum_ext.sum_cuda(a, b)

torch.testing.assert_close(out, a + b)
print("Success, same result!")
```

You can imagine how PyTorch works by overloading the `*` operator for tensors. 

Any time you do `a * b` on a tensor, PyTorch will call the `__mul__` method of the `Tensor` class. 

If a GPU is available, it will call the corresponding CUDA kernel, otherwise it will default to the CPU implementation.

Pseudocode to illustrate the process:

```python

import torch
import my_kernel

class CustomTensor(torch.Tensor):
    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            # Ensure both tensors are on the same device
            assert self.device == other.device, "Tensors must be on the same device"
            
            # Allocate output tensor on the correct device
            out = torch.zeros_like(self)

            # Check if running on GPU
            if self.is_cuda:
                my_kernel.multiply_kernel(self, other, out, self.numel())
            else:
                out = self.cpu().numpy() * other.cpu().numpy()  # Fallback to CPU
            
            return out
        else:
            return super().__mul__(other)
```

Hence the steps taken by the Python runtime are:
1. `__mul__` is called on the custom tensor.
2. If the other operand is a tensor, the `__mul__` method of the `Tensor` class is called.
3. If a GPU is available, the CUDA kernel is called.
4. The result is returned to the Python runtime.