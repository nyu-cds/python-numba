---
title: "Compilation Modes"
teaching: 10
exercises: 10
questions:
- "Is it possible to ensure Numba will generate high performance code?"
objectives:
- "Learn the difference between Numba's compilation modes."
- "See what happens if Numba can't generate optimized code."
keypoints:
- "The `nopython` mode will generate the best performance, but has limitations."
- "Types that can't be inferred by the compiler will generate an error."
---
Numba has two compilation modes: `nopython` mode and `object` mode. 

In `nopython` mode, the Numba compiler will generate code that 
does not access the Python C API. This mode produces the highest performance code, but requires that the native types of all 
values in the function can be inferred. 

In `object` mode, the Numba compiler generates code that handles all values as Python 
objects and uses the Python C API to perform all operations on those objects. Code compiled in `object` mode will often run 
no faster than Python interpreted code. 

Numba will by default automatically use `object` mode if `nopython` mode cannot be used for 
some reason. Rather than fall back to `object` mode, it is sometimes preferrable to generate an error instead. By adding the 
`nopython=True` keyword, it is possible to force Numbe to do this:

~~~
from numba import jit
@jit("void(f4[:])",nopython=True)
def bubblesort(X):
    N = len(X)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
~~~
{: .python}

Notice that this code compiles cleanly. However, if we introduce an object who's type cannot be inferred and see what 
happens (don't worry, you should see errors).

~~~
from numba import jit
from decimal import Decimal
@jit("void(f4[:])",nopython=True)
def bubblesort(X):
    N = len(X)
    val = Decimal(100)
    for end in range(N, 1, -1):
        for i in range(end - 1):
            cur = X[i]
            if cur > X[i + 1]:
                tmp = X[i]
                X[i] = X[i + 1]
                X[i + 1] = tmp
~~~
{: .python}
           
Now when we try to compile this code, Numba complains that `Decimal` is an untyped name. Without the `nopython` mode, this code would 
have compiled, but would have run much more slowly.

> ## Challenge
> Copy this code and remove the `nopython` option. Verify that it compiles cleanly in this case.
{: .challenge}
