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

#### 2.1.1 张量结构操作

张量的操作主要包括张量的**结构操作**和张量的**数学运算**。

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

本篇我们介绍张量的结构操作。

##### 2.1.1.1 创建张量

```python
a = torch.tensor([1,2,3],dtype = torch.float)

#均匀随机分布
torch.manual_seed(0)
minval,maxval = 0,10
a = minval + (maxval-minval)*torch.rand([5])

#正态分布随机
b = torch.normal(mean = torch.zeros(3,3), std = torch.ones(3,3))

#正态分布随机
mean,std = 2,5
c = std*torch.randn((3,3))+mean

#整数随机排列
d = torch.randperm(20)
```

##### 2.1.1.2 索引切片

张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。

可以通过索引和切片对部分元素进行修改。

此外，对于不规则的切片提取,可以使用torch.index_select, torch.masked_select, torch.take

如果要通过修改张量的某些元素得到新的张量，可以使用torch.where,torch.masked_fill,torch.index_fill

```python
#抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
# scores dim [4, 10, 7]
torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))

#抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
#take将输入看成一维数组，输出和index同形状
s = torch.take(scores,torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))

#抽取分数大于等于80分的分数（布尔索引）
#结果是1维张量
g = torch.masked_select(scores,scores>=80)
```

如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where,torch.index_fill 和 torch.masked_fill

torch.where可以理解为if的张量版本。

torch.index_fill的选取元素逻辑和torch.index_select相同。

torch.masked_fill的选取元素逻辑和torch.masked_select相同。

##### 2.1.1.3 维度变换

维度变换相关函数主要有 torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose

torch.reshape 可以改变张量的形状or张量的view的方法。

torch.squeeze 可以减少维度。

torch.unsqueeze 可以增加维度。

torch.transpose 可以交换维度。

##### 2.1.1.4 合并分割

可以用torch.cat方法和torch.stack方法将多个张量合并，可以用torch.split方法把一个张量分割成多个张量。

torch.cat和torch.stack有略微的区别，torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度。

torch.split是torch.cat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

#### 2.1.2 张量数学运算

标量运算，向量运算，矩阵运算和广播机制

##### 2.1.2.1 标量运算

加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。

标量运算符的特点是对张量实施**逐元素运算**。

有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。

```python
a = torch.tensor([[1.0,2],[-3,4.0]])
b = torch.tensor([[5.0,6],[7.0,8.0]])

a+b 

a-b

a/b

torch.clamp(a)

x = torch.tensor([2.6,-2.7])
print(torch.round(x)) #保留整数部分，四舍五入
print(torch.floor(x)) #保留整数部分，向下归整
print(torch.ceil(x))  #保留整数部分，向上归整
print(torch.trunc(x)) #保留整数部分，向0归整
```

##### 2.1.2.2 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。

```python
#统计值
a = torch.arange(1,10).float()
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a)) #累乘
print(torch.std(a))  #标准差
print(torch.var(a))  #方差
print(torch.median(a)) #中位数

#torch.sort和torch.topk可以对张量排序
a = torch.tensor([[9,7,8],[1,3,2],[5,6,4]]).float()
print(torch.topk(a,2,dim = 0),"\n")
print(torch.topk(a,2,dim = 1),"\n")
print(torch.sort(a,dim = 1),"\n")
```

##### 2.1.2.3 矩阵运算

矩阵必须是二维的。类似torch.tensor([1,2,3])这样的不是矩阵。

矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。

```python
#矩阵乘法
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[2,0],[0,2]])
print(a@b)  #等价于torch.matmul(a,b) 或 torch.mm(a,b)

#矩阵逆，必须为浮点类型
a = torch.tensor([[1.0,2],[3,4]])
print(torch.inverse(a))

#矩阵求trace
a = torch.tensor([[1.0,2],[3,4]])
print(torch.trace(a))

#矩阵求范数
a = torch.tensor([[1.0,2],[3,4]])
print(torch.norm(a))
```

##### 2.1.2.4 广播机制

Pytorch的广播规则和numpy是一样的:

- 1、如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
- 2、如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
- 3、如果两个张量在所有维度上都是相容的，它们就能使用广播。
- 4、广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
- 5、在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

torch.broadcast_tensors可以将多个张量根据广播规则转换成相同的维度。

#### 2.1.3  nn.functional和nn.Module

Pytorch和神经网络相关的功能组件大多都封装在 torch.nn模块下。

这些功能组件的绝大部分既有函数形式实现，也有类形式实现。

其中nn.functional(一般引入后改名为F)有各种功能组件的函数实现。例如：

(激活函数)

- F.relu
- F.sigmoid
- F.tanh
- F.softmax

(模型层)

- F.linear
- F.conv2d
- F.max_pool2d
- F.dropout2d
- F.embedding

(损失函数)

- F.binary_cross_entropy
- F.mse_loss
- F.cross_entropy

为了便于对参数进行管理，一般通过继承 nn.Module 转换成为类的实现形式，并直接封装在 nn 模块下。例如：

(激活函数)

- nn.ReLU
- nn.Sigmoid
- nn.Tanh
- nn.Softmax

(模型层)

- nn.Linear
- nn.Conv2d
- nn.MaxPool2d
- nn.Dropout2d
- nn.Embedding

(损失函数)

- nn.BCELoss
- nn.MSELoss
- nn.CrossEntropyLoss

实际上nn.Module除了可以管理其引用的各种参数，还可以管理其引用的子模块，功能十分强大。

##### 2.1.3.1 使用nn.Module来管理参数

在Pytorch中，模型的参数是需要被优化器训练的，因此，通常要设置参数为 requires_grad = True 的张量。

同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情。

Pytorch一般将参数用nn.Parameter来表示，并且用nn.Module来管理其结构下的所有参数。

一般情况下，我们都很少直接使用 nn.Parameter来定义参数构建模型，而是通过一些拼装一些常用的模型层来构造模型。

这些模型层也是继承自nn.Module的对象,本身也包括参数，属于我们要定义的模块的子模块。

nn.Module提供了一些方法可以管理这些子模块。

- children() 方法: 返回生成器，包括模块下的所有子模块。
- named_children()方法：返回一个生成器，包括模块下的所有子模块，以及它们的名字。
- modules()方法：返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身。
- named_modules()方法：返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身。

**其中chidren()方法和named_children()方法较多使用。**

modules()方法和named_modules()方法较少使用，其功能可以通过多个named_children()的嵌套使用实现。

### 2.2 中阶API（模型之墙）

我们将主要介绍Pytorch的如下中阶API

- 数据管道
- 模型层
- 损失函数
- TensorBoard可视化

如果把模型比作一个房子，那么中阶API就是【模型之墙】。

#### 2.2.1 DataSet和DataLoader



#### 2.2.2 模型层



#### 2.2.3 损失函数



#### 2.2.4 TensorBoard

Pytorch中利用TensorBoard可视化的大概过程如下：

首先在Pytorch中指定一个目录创建一个torch.utils.tensorboard.SummaryWriter日志写入器。

然后根据需要可视化的信息，利用日志写入器将相应信息日志写入我们指定的目录。

最后就可以传入日志目录作为参数启动TensorBoard，然后就可以在TensorBoard中愉快地看片了。

我们主要介绍Pytorch中利用TensorBoard进行如下方面信息的可视化的方法。

- 可视化模型结构： writer.add_graph
- 可视化指标变化： writer.add_scalar
- 可视化参数分布： writer.add_histogram
- 可视化原始图像： writer.add_image 或 writer.add_images
- 可视化人工绘图： writer.add_figure

##### 2.2.4.1 可视化模型结构

```python
import torch 
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model,summary
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)
```

```python
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(1,3,32,32))
writer.close()
```

```python
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard
from tensorboard import notebook

#查看启动的tensorboard程序
notebook.list() 

#启动tensorboard程序
notebook.start("--logdir ./data/tensorboard")
#等价于在命令行中执行 tensorboard --logdir ./data/tensorboard
#可以在浏览器中打开 http://localhost:6006/ 查看
```

##### 2.2.4.2 可视化指标

有时候在训练过程中，如果能够实时动态地查看loss和各种metric的变化曲线，那么无疑可以帮助我们更加直观地了解模型的训练情况。

> `writer.add_scalar`仅能对标量的值的变化进行可视化。因此它一般用于对loss和metric的变化进行可视化分析。

```python
import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter



# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

writer = SummaryWriter('./data/tensorboard')
for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    writer.add_scalar("x",x.item(),i) #日志中记录x在第step i 的值
    writer.add_scalar("y",y.item(),i) #日志中记录y在第step i 的值

writer.close()
    
print("y=",f(x).data,";","x=",x.data)
```

##### 2.2.4.3 可视化参数分布

如果需要对模型的参数(一般非标量)在训练过程中的变化进行可视化，可以使用 writer.add_histogram。

它能够观测张量值分布的直方图随训练步骤的变化趋势。

```python
import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter


# 创建正态分布的张量模拟参数矩阵
def norm(mean,std):
    t = std*torch.randn((100,20))+mean
    return t

writer = SummaryWriter('./data/tensorboard')
for step,mean in enumerate(range(-10,10,1)):
    w = norm(mean,1)
    writer.add_histogram("w",w, step)
    writer.flush()
writer.close()
```

##### 2.2.4.4 可视化原始图像

如果我们做图像相关的任务，也可以将原始的图片在tensorboard中进行可视化展示。

如果只写入一张图片信息，可以使用writer.add_image。

如果要写入多张图片信息，可以使用writer.add_images。

也可以用 torchvision.utils.make_grid将多张图片拼成一张图片，然后用writer.add_image写入。

注意，传入的是代表图片信息的Pytorch中的张量数据。

##### 2.2.4.5 可视化人工绘图

如果我们将matplotlib绘图的结果再 tensorboard中展示，可以使用 add_figure.

注意，和writer.add_image不同的是，writer.add_figure需要传入matplotlib的figure对象。

### 2.3 高阶API（模型之屋）

Pytorch没有官方的高阶API。一般通过nn.Module来构建模型并编写自定义训练循环。

为了更加方便地训练模型，作者编写了仿keras的Pytorch模型接口：torchkeras， 作为Pytorch的高阶API。

本章我们主要详细介绍Pytorch的高阶API如下相关的内容。

- 构建模型的3种方法(继承nn.Module基类，使用nn.Sequential，辅助应用模型容器)
- 训练模型的3种方法(脚本风格，函数风格，torchkeras.Model类风格)
- 使用GPU训练模型(单GPU训练，多GPU训练)

#### 2.3.1 构建模型的3种方法

可以使用以下3种方式构建模型：

1，继承nn.Module基类构建自定义模型。

2，使用nn.Sequential按层顺序构建模型。

3，继承nn.Module基类构建模型并辅助应用模型容器进行封装(nn.Sequential,nn.ModuleList,nn.ModuleDict)。

其中 第1种方式最为常见，第2种方式最简单，第3种方式最为灵活也较为复杂。

推荐使用第1种方式构建模型。

##### 2.3.1.1 继承nn.Module基类构建自定义模型

以下是继承nn.Module基类构建自定义模型的一个范例。模型中的用到的层一般在`__init__`函数中定义，然后在`forward`方法中定义模型的正向传播逻辑。

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)
```

##### 2.3.1.2 使用nn.Sequential按层顺序构建模型

使用nn.Sequential按层顺序构建模型无需定义forward方法。仅仅适合于简单的模型。

以下是使用nn.Sequential搭建模型的一些等价方法。

- 使用add_module方法

  ```python
  net = nn.Sequential()
  net.add_module("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3))
  net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
  net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
  net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
  net.add_module("dropout",nn.Dropout2d(p = 0.1))
  net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
  net.add_module("flatten",nn.Flatten())
  net.add_module("linear1",nn.Linear(64,32))
  net.add_module("relu",nn.ReLU())
  net.add_module("linear2",nn.Linear(32,1))
  net.add_module("sigmoid",nn.Sigmoid())
  ```

- 利用变长参数

  这种方式构建时不能给每个层指定名称。

  ```python
  net = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
      nn.MaxPool2d(kernel_size = 2,stride = 2),
      nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
      nn.MaxPool2d(kernel_size = 2,stride = 2),
      nn.Dropout2d(p = 0.1),
      nn.AdaptiveMaxPool2d((1,1)),
      nn.Flatten(),
      nn.Linear(64,32),
      nn.ReLU(),
      nn.Linear(32,1),
      nn.Sigmoid()
  )
  ```

- 利用OrderedDict

  ```python
  from collections import OrderedDict
  
  net = nn.Sequential(OrderedDict(
            [("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)),
              ("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2)),
              ("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)),
              ("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2)),
              ("dropout",nn.Dropout2d(p = 0.1)),
              ("adaptive_pool",nn.AdaptiveMaxPool2d((1,1))),
              ("flatten",nn.Flatten()),
              ("linear1",nn.Linear(64,32)),
              ("relu",nn.ReLU()),
              ("linear2",nn.Linear(32,1)),
              ("sigmoid",nn.Sigmoid())
            ])
          )
  print(net)
  ```

##### 2.3.1.3 继承nn.Module基类构建模型并辅助应用模型容器进行封装

当模型的结构比较复杂时，我们可以应用模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)对模型的部分结构进行封装。

这样做会让模型整体更加有层次感，有时候也能减少代码量。

>  Note：在下面的范例中我们每次仅仅使用一种模型容器，但实际上这些模型容器的使用是非常灵活的，可以在一个模型中任意组合任意嵌套使用。

- nn.Sequential作为模型容器

  ```python
  class Net(nn.Module):
      
      def __init__(self):
          super(Net, self).__init__()
          self.conv = nn.Sequential(
              nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
              nn.MaxPool2d(kernel_size = 2,stride = 2),
              nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
              nn.MaxPool2d(kernel_size = 2,stride = 2),
              nn.Dropout2d(p = 0.1),
              nn.AdaptiveMaxPool2d((1,1))
          )
          self.dense = nn.Sequential(
              nn.Flatten(),
              nn.Linear(64,32),
              nn.ReLU(),
              nn.Linear(32,1),
              nn.Sigmoid()
          )
      def forward(self,x):
          x = self.conv(x)
          y = self.dense(x)
          return y 
      
  net = Net()
  print(net)
  ```

- nn.ModuleList作为模型容器

  ```python
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.layers = nn.ModuleList([
              nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
              nn.MaxPool2d(kernel_size = 2,stride = 2),
              nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
              nn.MaxPool2d(kernel_size = 2,stride = 2),
              nn.Dropout2d(p = 0.1),
              nn.AdaptiveMaxPool2d((1,1)),
              nn.Flatten(),
              nn.Linear(64,32),
              nn.ReLU(),
              nn.Linear(32,1),
              nn.Sigmoid()]
          )
      def forward(self,x):
          for layer in self.layers:
              x = layer(x)
          return x
  net = Net()
  print(net)
  ```

  > Note：上面中的ModuleList不能用Python中的列表代替。

- nn.ModuleDict作为模型容器

  ```python
  class Net(nn.Module):
      
      def __init__(self):
          super(Net, self).__init__()
          self.layers_dict = nn.ModuleDict({"conv1":nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
                 "pool": nn.MaxPool2d(kernel_size = 2,stride = 2),
                 "conv2":nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
                 "dropout": nn.Dropout2d(p = 0.1),
                 "adaptive":nn.AdaptiveMaxPool2d((1,1)),
                 "flatten": nn.Flatten(),
                 "linear1": nn.Linear(64,32),
                 "relu":nn.ReLU(),
                 "linear2": nn.Linear(32,1),
                 "sigmoid": nn.Sigmoid()
                })
      def forward(self,x):
          layers = ["conv1","pool","conv2","pool","dropout","adaptive",
                    "flatten","linear1","relu","linear2","sigmoid"]
          for layer in layers:
              x = self.layers_dict[layer](x)
          return x
  net = Net()
  print(net)
  ```

  > Note：上面中的ModuleDict不能用Python中的字典代替。

#### 2.3.2 训练模型的三种方法

有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。

- 脚本形式训练循环

  整个训练过程完全自定义，较为繁琐，难以迁移应用

- 函数式训练循环

  基于脚本训练的循环的过程，做了一些函数的封装，一定程度上可以迁移应用，但是不够灵活

- 类形式封装

  构建完成模型后，可以直接调用，灵活且易迁移，方便使用。但是对于一些复杂模型来说，可能不易使用，需要和函数式训练循环结合使用。

#### 2.3.3 使用GPU训练模型

Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上。核心代码只有以下几行。

- 单GPU训练

  ```python
  # 定义模型
  ... 
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device) # 移动模型到cuda
  
  # 训练模型
  ...
  
  features = features.to(device) # 移动数据到cuda
  labels = labels.to(device) # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels
  ...
  ```

- 多GPU训练

  如果要使用多个GPU训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。 则模型移动到GPU上之后，会在每一个GPU上拷贝一个副本，并把数据平分到各个GPU上进行训练。核心代码如下，

  ```python
  # 定义模型
  ... 
  
  if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model) # 包装为并行风格模型
  
  # 训练模型
  ...
  features = features.to(device) # 移动数据到cuda
  labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
  ...
  ```

##### 2.3.3.1 GPU基本操作

```python
# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)
if_cuda= True
gpu_count= 1
# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # 或者 tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu() 
print(tensor_cpu.device)
cuda:0
True
cpu
# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)
False
True
cuda:0
# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "./data/model_parameter.pkl") 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl")) 
cpu
[0]
cuda:0
# 5，清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()
```

