---
layout: lesson
root: .
---
Numba provides the ability to speed up applications with high performance functions written directly in Python, rather than using language 
extensions such as Cython.

Numba allows the compilation of selected portions of pure Python code to native code, and generates optimized machine code using the 
[LLVM](http://llvm.org/) compiler infrastructure. 

With a few simple annotations, array-oriented and math-heavy Python code can be just-in-time (JIT) optimized to achieve 
performance similar to C, C++ and Fortran, without having to switch languages or Python interpreters. 

Numba works at the function level. From a function, Numba can generate native code for that function as well as the wrapper code needed to 
call it directly from Python. This compilation is done on-the-fly and in-memory.

Numba’s main features are:
* On-the-fly code generation (at import time or runtime, at the user’s preference)
* Native code generation for the CPU (default) and GPU hardware
* Integration with the Python scientific software stack (thanks to NumPy)

> ## Prerequisites
>
> The examples in this lesson can be run directly using the Python interpreter, using IPython interactively, 
> or using Jupyter notebooks. Anaconda users will already have Cython installed. You will also need a functioning
> C compiler to be able to use Cython. See the [Cython installation guide](http://cython.readthedocs.io/en/latest/src/quickstart/install.html) for more details.
{: .prereq}

