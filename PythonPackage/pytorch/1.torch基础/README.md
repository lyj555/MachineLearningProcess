## 1. pytorch 核心概念

Pytorch是一个基于Python的机器学习库。它广泛应用于计算机视觉，自然语言处理等深度学习领域。是目前和TensorFlow分庭抗礼的深度学习框架，在学术圈颇受欢迎。

它主要提供了以下两种核心功能：

1，支持GPU加速的张量计算。

2，方便优化模型的自动微分机制。

Pytorch的主要优点：

- 简洁易懂：Pytorch的API设计的相当简洁一致。基本上就是tensor, autograd, nn三级封装。学习起来非常容易。有一个这样的段子，说TensorFlow的设计哲学是 Make it complicated, Keras 的设计哲学是 Make it complicated and hide it, 而Pytorch的设计哲学是 Keep it simple and stupid.
- 便于调试：Pytorch采用动态图，可以像普通Python代码一样进行调试。不同于TensorFlow, Pytorch的报错说明通常很容易看懂。有一个这样的段子，说你永远不可能从TensorFlow的报错说明中找到它出错的原因。
- 强大高效：Pytorch提供了非常丰富的模型组件，可以快速实现想法。并且运行速度很快。目前大部分深度学习相关的Paper都是用Pytorch实现的。

### 1.1 张量数据结构

#### 1.1.1 张量数据类型

张量的数据类型和numpy.array基本一一对应，但是不支持str类型。

包括:

torch.float64(torch.double),

**torch.float32(torch.float)**,

torch.float16,

torch.int64(torch.long),

torch.int32(torch.int),

torch.int16,

torch.int8,

torch.uint8,

torch.bool

#### 1.1.2 张量的维度

不同类型的数据可以用不同维度(dimension)的张量来表示。

标量为0维张量，向量为1维张量，矩阵为2维张量。

彩色图像有rgb三个通道，可以表示为3维张量。

视频还有时间维，可以表示为4维张量。

可以简单地总结为：有几层中括号，就是多少维的张量。

#### 1.1.3 张量的尺寸

可以使用 shape属性或者 size()方法查看张量在每一维的长度.

可以使用view方法改变张量的尺寸。

如果view方法改变尺寸失败，可以使用reshape方法.

#### 1.1.4 张量和numpy数组

可以用numpy方法从Tensor得到numpy数组，也可以用torch.from_numpy从numpy数组得到Tensor。

这两种方法关联的Tensor和numpy数组是共享数据内存的。

如果改变其中一个，另外一个的值也会发生改变。

如果有需要，可以用张量的clone方法拷贝张量，中断这种关联。

此外，还可以使用item方法从标量张量得到对应的Python数值。

使用tolist方法从张量得到对应的Python数值列表。

### 1.2 动态计算图



### 1.3 自动微分机制



## 2. Pytorch的层次结构

Pytorch的层次结构从低到高可以分成如下五层。

最底层为硬件层，Pytorch支持CPU、GPU加入计算资源池。

第二层为C++实现的内核。

第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、自动微分、变量管理. 如torch.tensor,torch.cat,torch.autograd.grad,nn.Module. 如果把模型比作一个房子，那么第三层API就是【模型之砖】。

第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道等等。 如torch.nn.Linear,torch.nn.BCE,torch.optim.Adam,torch.utils.data.DataLoader. 如果把模型比作一个房子，那么第四层API就是【模型之墙】。

第五层为Python实现的模型接口。Pytorch没有官方的高阶API。为了便于训练模型，作者仿照keras中的模型接口，使用了不到300行代码，封装了Pytorch的高阶模型接口torchkeras.Model。如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。

### 2.1 低阶API（模型之砖）

Pytorch的低阶API主要包括张量操作，动态计算图和自动微分。

如果把模型比作一个房子，那么低阶API就是【模型之砖】。

在低阶API层次上，可以把**Pytorch当做一个增强版的numpy来使用**。

Pytorch提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。

前面几章我们对低阶API已经有了一个整体的认识，本章我们将重点详细介绍张量操作和动态计算图。

张量的操作主要包括张量的结构操作和张量的数学运算。

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

动态计算图我们将主要介绍动态计算图的特性，计算图中的Function，计算图与反向传播。

### 2.2 中阶API（模型之墙）



### 2.3 高阶API（模型之屋）