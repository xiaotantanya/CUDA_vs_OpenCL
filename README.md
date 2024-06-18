# CUDA与Opencl在GPU上性能对比

# 实验环境

* 操作系统: Ubuntu18.04
* CPU: Intel(R) Core(TM) i9-9900X CPU @ 3.50GHz
* GPU: Nvidia 2080Ti（显存10GB）
* 内存: 64GB
* CUDA Toolkit 11.4
* OpenCL 3.0

# 实验

## 简单运算操作
在该测试中，我们只记录在GPU上运行的时间，不包括CUDA或者Opencl初始化和最后内存释放所花时间，注意其中使用chrono库来记录运行时间。每次测试中运行五次，记录所花时间中位数。

### Add&Sub 操作
在加减法测试中，我们定义了两个长度相同的单精度浮点数数组，并分别初始化为1.0f，2.0f，然后进行加减法操作，测试结果如下，其中每列代表在不同平台下，进行n次加减法操作所花时间（单位为ms）。
* 加法操作

|  平台  |   1000   |  10000   |  100000  | 1000000 | 10000000 | 100000000 |
| :----: | :------: | :------: | :------: | :-----: | :------: | :-------: |
|  CUDA  | 0.016998 | 0.017241 |  0.0393  | 0.08899 | 0.296701 |  2.22198  |
| OpenCL | 0.198947 | 0.225979 | 0.495549 | 2.7485  | 25.9428  |  260.319  |
|  CPU   | 0.009822 | 0.116976 | 1.20946  | 7.77397 | 36.4262  |  347.493  |

* 减法操作

|  平台  |   1000   |  10000   |  100000  | 1000000  | 10000000 | 100000000 |
| :----: | :------: | :------: | :------: | :------: | :------: | :-------: |
|  CUDA  | 0.01963  | 0.017222 | 0.039401 | 0.092633 | 0.294654 |  2.22506  |
| OpenCL | 0.193207 | 0.224084 | 0.498801 | 2.82131  | 26.7663  |  252.961  |
|  CPU   | 0.009106 | 0.11097  | 1.13158  | 7.05745  | 35.884   |  337.065  |

### Multiply&Devide 操作
在乘除法测试中，我们定义了两个长度相同的单精度浮点数数组，并分别初始化为1.0f，2.0f，然后进行乘除法操作，测试结果如下，其中每列代表在不同平台下，进行n次乘除法操作所花时间（单位为ms）。
* 乘法操作

|  平台  |   1000   |  10000   |  100000  | 1000000  | 10000000 | 100000000 |
| :----: | :------: | :------: | :------: | :------: | :------: | :-------: |
|  CUDA  | 0.019717 | 0.019464 | 0.041503 | 0.091946 | 0.298064 |  2.22754  |
| OpenCL | 0.193972 | 0.226775 | 0.492243 | 2.72615  | 26.8028  |  254.514  |
|  CPU   |  0.0101  | 0.116718 | 1.21057  | 7.60248  |  36.92   |  350.485  |

* 除法操作

|  平台  |   1000   |  10000   |  100000  | 1000000  | 10000000 | 100000000 |
| :----: | :------: | :------: | :------: | :------: | :------: | :-------: |
|  CUDA  | 0.018636 | 0.019924 | 0.040569 | 0.087888 | 0.295593 |  2.22387  |
| OpenCL | 0.197692 | 0.222616 | 0.515666 | 2.80643  | 26.4215  |  264.745  |
|  CPU   | 0.009487 | 0.116505 | 1.15818  | 7.20175  | 36.7899  |  347.824  |

## 矩阵运算
在矩阵运算中，我们假设矩阵都是nxn的矩阵，并使用随机初始化矩阵中每位单精度浮点数，随机初始化范围为1~11，测试结果如下，其中每列代表在不同平台下，进行矩阵维度为nxn时，矩阵乘法操作所花时间（单位为ms），其中CPU在矩阵维度为4096和8192跑所花时间太长，只跑了一次。

|  平台  |   256   |   512   |  1024   |  2048   |    4096    |    8192    |
| :----: | :-----: | :-----: | :-----: | :-----: | :--------: | :--------: |
|  CUDA  | 0.29984 | 2.27251 | 17.9936 | 105.488 |  859.491   |  7276.15   |
| OpenCL | 0.8616  | 4.40746 | 28.2758 | 152.938 |  1601.72   |  54061.2   |
|  CPU   | 48.6866 | 575.389 | 5593.99 | 60821.3 | 1.12099e+06| very large |

## FFT运算
FFT运算是比较复杂的操作，CUDA直接使用cuFFT来实现傅里叶变换。OpenCL并没有直接类似于cuFFT的库来进行傅里叶变换，所以需要手动编写OpenCL内核来实现傅里叶变换。这里不具体讲解傅里叶变换内部实现和实际意义。在测试中，其中数据初始化为1+0i, NUM_TRANSFORMS代表要执行的 FFT 变换的总数量，TRANSFORM_SIZE代表单个傅里叶变换的大小，该次测试将TRANSFORM_SIZE设置为2048。在不同平台下，针对不同NUM_TRANSFORMS，记录傅里叶变换操作所花时间（单位为ms）。

**待实现**

## 总结
综合来看，基于CUDA实现的代码在2080TiGPU上运行更快，有着几倍的差距，但目前并没有在AMD显卡或者其他品牌的显卡上测试。且CUDA是专门为Nvidia显卡设计的，性能比基于Opencl实现的代码效率更好也是有道理的。同时，Opencl的一大特点是可移植性好，代码可以在多个平台以及嵌入式设备上使用，在本次实验中并没有进行测试。在本次实验中也发现，基于CUDA的编程较为简单，理解也较Opencl简单，同时网上资料也较多。而Opencl为了兼容性，实现比较偏底层，编程需要遵守流程，网上资料也偏少。如果只是使用Nvidia的显卡进行GPU编程，且不需要多平台兼容的话，使用CUDA能有更高的效率，编程也更加容易。


## 注意
目前由于ubuntu18.04版本较老，当我使用clinfo输出opencl版本信息时，会出现提示：
```bash
NOTE:   your OpenCL library only supports OpenCL 2.2,
                but some installed platforms support OpenCL 3.0.
                Programs using 3.0 features may crash
                or behave unexepectedly
```
网上查询这个提示，有些[网页](https://stackoverflow.com/questions/69971216/how-to-resolve-the-mismatch-between-opencl-library-and-opencl-platform)说可能需要更新Intel Compute Engine，但测试服务器并不是我一个人的，我怕更新后会影响到其他人，就没有弄。实际测试过程中，OpenCL也使用GPU，这可能类似与一个警告而不是错误。
