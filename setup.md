---
layout: page
title: Setup
permalink: /setup/
---
Numba is part of the Anaconda Python distribution. If Number is not already installed, you can install it manually using the command below.

~~~
conda install numba
~~~
{: .bash}

## CUDA Support

Numba supports CUDA-enabled GPUs with compute capability 2.0 or above with an up-to-data Nvidia driver. You can still use Numba without a 
CUDA-enabled GPU using the CUDA simulator, however you will not get the performance benefits of running on a GPU.

In order to use the Numba CUDA support you will need the CUDA toolkit installed. Since you are using Anaconda, just type:

~~~
$ conda install cudatoolkit
~~~
{: .bash}

It is also possible to install the CUDA Toolkit directly from NVIDIA, however this may require additional configuration to enable it to work 
with Numba so is not recommended.
