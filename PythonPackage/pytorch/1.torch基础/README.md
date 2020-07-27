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

### 1.2 自动微分机制

Pytorch一般通过反向传播 backward 方法 实现这种求梯度计算。该方法求得的梯度将存在对应自变量张量的grad属性下。

除此之外，也能够调用torch.autograd.grad 函数来实现求梯度计算。

这就是Pytorch的自动微分机制。

#### 1.2.1 利用backward方法求导数

- 标量反向传播

  ```python
  import numpy as np 
  import torch 
  
  # f(x) = a*x**2 + b*x + c的导数
  
  x = torch.tensor(0.0,requires_grad = True) # x需要被求导
  a = torch.tensor(1.0)
  b = torch.tensor(-2.0)
  c = torch.tensor(1.0)
  y = a*torch.pow(x,2) + b*x + c 
  
  y.backward()
  dy_dx = x.grad
  print(dy_dx)
  ```

  out：tensor(-2.)

- 非标量反向传播

  ```python
  import numpy as np 
  import torch 
  
  # f(x) = a*x**2 + b*x + c
  
  x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
  a = torch.tensor(1.0)
  b = torch.tensor(-2.0)
  c = torch.tensor(1.0)
  y = a*torch.pow(x,2) + b*x + c 
  
  gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
  
  print("x:\n",x)
  print("y:\n",y)
  y.backward(gradient = gradient)
  x_grad = x.grad
  print("x_grad:\n",x_grad)
  ```

  out：

  ```
  x:
   tensor([[0., 0.],
          [1., 2.]], requires_grad=True)
  y:
   tensor([[1., 1.],
          [0., 1.]], grad_fn=<AddBackward0>)
  x_grad:
   tensor([[-2., -2.],
          [ 0.,  2.]])
  ```

- **非标量的反向传播可以用标量的反向传播实现**

  ```python
  import numpy as np 
  import torch 
  
  # f(x) = a*x**2 + b*x + c
  
  x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
  a = torch.tensor(1.0)
  b = torch.tensor(-2.0)
  c = torch.tensor(1.0)
  y = a*torch.pow(x,2) + b*x + c 
  
  gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
  z = torch.sum(y*gradient)
  
  print("x:",x)
  print("y:",y)
  z.backward()
  x_grad = x.grad
  print("x_grad:\n",x_grad)
  ```

  out：

  ```
  x: tensor([[0., 0.],
          [1., 2.]], requires_grad=True)
  y: tensor([[1., 1.],
          [0., 1.]], grad_fn=<AddBackward0>)
  x_grad:
   tensor([[-2., -2.],
          [ 0.,  2.]])
  ```

#### 1.2.2 利用autograd.grad方法求导数

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的导数

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c


# create_graph 设置为 True 将允许创建更高阶的导数 
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx,x)[0] 

print(dy2_dx2.data)
```

out：

```
tensor(-2.)
tensor(2.)
```

#### 1.2.3 利用自动微分和优化器求最小值

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的最小值

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
   
    
print("y=",f(x).data,";","x=",x.data)
```

out：

```
y= tensor(0.) ; x= tensor(1.0000)
```

### 1.3 动态计算图

本节我们将介绍 Pytorch的动态计算图。

包括：

- 动态计算图简介
- 计算图中的Function
- 计算图和反向传播
- 叶子节点和非叶子节点
- 计算图在TensorBoard中的可视化

#### 1.3.1 动态计算图简介

Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。

Pytorch中的计算图是动态图。这里的动态主要有两重含义。

第一层含义是：**计算图的正向传播是立即执行的**。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。

第二层含义是：**计算图在反向传播后立即销毁**。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

**计算图在反向传播后立即销毁。**

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

#计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward()  #loss.backward(retain_graph = True) 

#loss.backward() #如果再次执行反向传播将报错
```

#### 1.3.2 计算图中的Function

计算图中的 张量我们已经比较熟悉了, 计算图中的另外一种节点是Function, 实际上就是 Pytorch中各种对张量操作的函数。

这些Function和我们Python中的函数有一个较大的区别，那就是**它同时包括正向计算逻辑和反向传播的逻辑**。

我们可以通过继承torch.autograd.Function来创建这种支持反向传播的Function。

#### 1.3.3 计算图与反向传播

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
```

loss.backward()语句调用后，依次发生以下计算过程：

1. loss自己的grad梯度赋值为1，即对自身的梯度为1。

2. loss根据其自身梯度以及关联的backward方法，计算出其对应的自变量即y1和y2的梯度，将该值赋值到y1.grad和y2.grad。

3. y2和y1根据其自身梯度以及关联的backward方法, 分别计算出其对应的自变量x的梯度，x.grad将其收到的多个梯度值累加。

> Note：1,2,3步骤的求梯度顺序和对多个梯度值的累加规则恰好是求导链式法则的程序表述）

**正因为求导链式法则衍生的梯度累加规则，张量的grad梯度不会自动清零，在需要的时候需要手动置零。**

#### 1.3.4 叶子节点和非叶子节点

执行下面代码，我们会发现 loss.grad并不是我们期望的1,而是 None。

类似地 y1.grad 以及 y2.grad也是 None.

这是为什么呢？这是由于它们不是叶子节点张量。

在反向传播过程中，只有 is_leaf=True 的叶子节点，需要求导的张量的导数结果才会被最后保留下来。

那么什么是叶子节点张量呢？叶子节点张量需要满足两个条件。

- 叶子节点张量是由用户直接创建的张量，而非由某个Function通过计算得到的张量。

- 叶子节点张量的 requires_grad属性必须为True.

**Pytorch设计这样的规则主要是为了节约内存或者显存空间，因为几乎所有的时候，用户只会关心他自己直接创建的张量的梯度。**

所有依赖于叶子节点张量的张量, 其requires_grad 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
print("loss.grad:", loss.grad)
print("y1.grad:", y1.grad)
print("y2.grad:", y2.grad)
print(x.grad)
```

out：

```
loss.grad: None
y1.grad: None
y2.grad: None
tensor(4.)
```

```python
print(x.is_leaf)
print(y1.is_leaf)
print(y2.is_leaf)
print(loss.is_leaf)
```

out：

```
True
False
False
False
```

如果需要保留中间计算结果的梯度到grad属性中，可以使用 retain_grad方法。 如果仅仅是为了调试代码查看梯度值，可以利用register_hook打印日志。

```python
import torch 

#正向传播
x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

#非叶子节点梯度显示控制
y1.register_hook(lambda grad: print('y1 grad: ', grad))
y2.register_hook(lambda grad: print('y2 grad: ', grad))
loss.retain_grad()

#反向传播
loss.backward()
print("loss.grad:", loss.grad)
print("x.grad:", x.grad)
```

out：

```
y2 grad:  tensor(4.)
y1 grad:  tensor(-4.)
loss.grad: tensor(1.)
x.grad: tensor(4.)
```

#### 1.3.5 计算图在TensorBoard中的可视化

可以利用 torch.utils.tensorboard 将计算图导出到 TensorBoard进行可视化。

```python
from torch import nn 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2,1))
        self.b = nn.Parameter(torch.zeros(1,1))

    def forward(self, x):
        y = x@self.w + self.b
        return y

net = Net()
```

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(10,2))
writer.close()
```

```python
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard
from tensorboard import notebook
notebook.list() 
#在tensorboard中查看模型
notebook.start("--logdir ./data/tensorboard")
```

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