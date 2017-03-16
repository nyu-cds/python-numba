---
title: "Compiling Code With `@jit`"
teaching: 20
exercises: 0
questions:
objectives:
keypoints:
---
Numba's central feature is the number.jit() decoration. Using this decorator, it is possible to mark a function for optimization by Numba’s 
JIT compiler. Various invocation modes trigger differing compilation options and behaviours.

Let's see Numba in action. The following is a Python implementation of bubblesort for NumPy arrays.

~~~
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

First we’ll create an array of sorted values and randomly shuffle them.

~~~
import numpy as np
​
original = np.arange(0.0, 10.0, 0.01, dtype='f4')
shuffled = original.copy()
np.random.shuffle(shuffled)
~~~
{: .python}

Next, create a copy and do a bubble sort on the copy.

~~~
sorted = shuffled.copy()
bubblesort(sorted)
print(np.array_equal(sorted, original))
~~~
{: .python}

When this is run, you should see the following:

~~~
True
~~~
{: .output}

Now let's time the execution. Note: we need to copy the array so we sort a random array each time as sorting an already sorted array is faster and
so would distort our timing.

~~~
%timeit sorted[:] = shuffled[:]; bubblesort(sorted)
10 loops, best of 3: 175 ms per loop
~~~
{: .python}

Ok, so we know how fast the pure Python implementation is. The recommended way to use the `@jit` decorator is to let Numba decide when and how to 
optimize, so we simply add the decorator to the function:

~~~
from numba import jit
@jit
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

Now we can check the execution time of the optimized code. 

~~~
%timeit sorted[:] = shuffled[:]; bubblesort(sorted)
The slowest run took 103.54 times longer than the fastest. This could mean that an intermediate result is being cached.
1000 loops, best of 3: 1.29 ms per loop
~~~
{: .python}

Using the decorator in this way will defer compilation until the first function execution, so the first execution will be significantly slower. 

Numba will infer the argument types at call time, and generate optimized code based on this 
information. Numba will also be able to compile separate specializations depending on the input types.

>## Function Signatures
>
> It is also possible to specify the signature of the Numba function. A function signature describes the types of the arguments and the return 
> type of the function. This can produce slightly faster code as the compiler does not need to infer the types. However the function is no 
> longer able to accept other types. See the numba.jit() documentation for more information on signatures. 
>
> For the sort function, this would be:
>
> ~~~
> from numba import jit
> @jit("void(f4[:])")
> def bubblesort(X):
>    N = len(X)
>    for end in range(N, 1, -1):
>        for i in range(end - 1):
>            cur = X[i]
>            if cur > X[i + 1]:
>                tmp = X[i]
>                X[i] = X[i + 1]
>                X[i + 1] = tmp
> ~~~
> {: .python}
>
> Time this code and see if it is any faster than the previous version.
{: .challenge}
