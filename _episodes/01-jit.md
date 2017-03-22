---
title: "Just-in-time Compiling"
teaching: 20
exercises: 0
questions:
- "How does Numba just-in-time compiling work?"
objectives:
- "Learn how to use the `@jit` decoration to improve performance."
keypoints:
- "The central feature of Numba is the `@jit` decoration."
---
Numba's central feature is the `numba.jit()` decoration. Using this decorator, it is possible to mark a function for optimization by Numba’s 
JIT compiler. Various invocation modes trigger differing compilation options and behaviours.

>## Python Decorators
> Decorators are a way to uniformly modify functions in a particular way. You can think of them as functions that take 
> functions as input and produce a function as output. See the 
> [Python reference documentation](https://docs.python.org/3/reference/compound_stmts.html#function-definitions) for a detailed discussion.
> 
> A function definition may be wrapped by one or more [decorator](http://docs.python.org/glossary.html#term-decorator) expressions. Decorator 
> expressions are evaluated when the function is defined, in the scope that contains the function definition. The result must be a callable, 
> which is invoked with the function object as the only argument. The returned value is bound to the function name instead of the function object. 
> Multiple decorators are applied in nested fashion. For example, the following code:
>
> ~~~
> @f1(arg) 
> @f2 
> def func(): 
>   pass
> ~~~
> {: .python}
> 
> is equivalent to:
> 
> ~~~
> def func(): 
>   pass 
> 
> func = f1(arg)(f2(func))
> ~~~
> {: .python}
>
> As pointed out there, they are not limited neccesarily to function definitions, and 
> [can also be used on class definitions](https://docs.python.org/3/reference/compound_stmts.html#class-definitions).
{: .callout}

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
