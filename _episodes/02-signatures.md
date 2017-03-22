---
title: "Function Signatures"
teaching: 10
exercises: 10
questions:
- "Is it possible to use function type information to improve performance with Numba?" 
objectives:
- "Learn how to specify function signatures."
- "Learn the different function signature notations."
keypoints:
- "Function signatures provide Numba with additional information that can help improve performance."
- "Specialization can make the functions less flexible."
---
It is also possible to specify the signature of the Numba function. A function signature describes the types of the arguments and the return 
type of the function. This can produce slightly faster code as the compiler does not need to infer the types. However the function is no 
longer able to accept other types. 

~~~
from numba import jit, int32, float64

@jit(float64(int32, int32))
def f(x, y):
    # A somewhat trivial example
    return (x + y) / 3.14
~~~
{: .python}

In this example, `float64(int32, int32)` is the function’s signature specifying a function that takes two 32-bit integer arguments and returns
a double precision float. Numba provides a [shorthand notation](http://numba.pydata.org/numba-doc/0.31.0/reference/types.html#numbers), so the same 
signature can be specified as `f8(i4, i4)`.

The specialization will be compiled by the `@jit` decorator, and no other specialization will be allowed. This is useful if you want fine-grained 
control over types chosen by the compiler (for example, to use single-precision floats).

If you omit the return type, e.g. by writing `(int32, int32)` instead of `float64(int32, int32)`, Numba will try to infer it for you. 
Function signatures can also be strings, and you can pass several of them as a list; see the `numba.jit()` documentation for more details.

Of course, the compiled function gives the expected results:

~~~
f(1, 3)
1.2738853503184713
~~~
{: .output}

Array signatures are specified by subscripting a base type according to the number of dimensions. For example a 1-dimension single-precision array 
would be written `float32[:]`, or a 3-dimension array of the same underlying type would be `f4[:,:,:]` (using the shorthand notation).

For the sort function we saw previously, the signature would be:

> ## Challenge
> What do you think the function signature for the `bubblesort` function would be? (Hint: the function does not return any value.)
>
> Add the funcion signature and try timing the code again. Is it any faster?
{: .challenge}
