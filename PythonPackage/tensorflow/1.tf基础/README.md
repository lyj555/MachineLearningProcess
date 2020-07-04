## 1. Tensorflow核心概念

### 1.1 张量的数据结构

程序 = 数据结构+算法。

TensorFlow程序 = 张量数据结构 + 计算图算法语言

张量和计算图是 TensorFlow的核心概念。

Tensorflow的基本数据结构是张量Tensor。张量即多维数组。Tensorflow的张量和numpy中的array很类似。

从行为特性来看，有两种类型的张量，**常量constant和变量Variable.**

常量的值在计算图中不可以被重新赋值，变量可以在计算图中用assign等算子重新赋值

- **常量**（constant）

  张量的数据类型和numpy.array基本一一对应。（不可以改变）

- **变量**（variable）

  模型中需要被训练的参数一般被设置成变量。（可以改变）

### 1.2 计算图

目前版本的tensorflow中有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。

#### 1.2.1 计算图简介

计算图由节点(nodes)和线(edges)组成。

节点表示操作符Operator，或者称之为算子，线表示计算间的依赖。

实线表示有数据传递依赖，传递的数据即张量。

虚线通常可以表示控制依赖，即执行先后顺序。

#### 1.2.2 静态计算图

在TensorFlow1.0中，使用静态计算图分两步，第一步定义计算图，第二步在会话中执行计算图。

TensorFlow2.0为了确保对老版本tensorflow项目的兼容性，在tf.compat.v1子模块中保留了对TensorFlow1.0那种静态计算图构建风格的支持。

> 创建一个session，显式地执行运行计算图

#### 1.2.3 动态计算图

动态计算图已经不区分计算图的定义和执行了，而是定义后立即执行。因此称之为 Eager Excution（类似numpy）。

使用动态计算图即Eager Excution的好处是方便调试程序，它会让TensorFlow代码的表现和Python原生代码的表现一样，写起来就像写numpy一样，各种日志打印，控制流全部都是可以使用的。

使用动态计算图的缺点是运行效率相对会低一些。因为使用动态图会有许多次Python进程和TensorFlow的C++进程之间的通信。而静态计算图构建完成之后几乎全部在TensorFlow内核上使用C++代码执行，效率更高。此外静态图会对计算步骤进行一定的优化，剪去和结果无关的计算步骤。

#### 1.2.4 AutoGraph

针对动态计算图中效率低的情况，做了相应的改进。

可以用@tf.function装饰器将普通Python函数转换成和TensorFlow1.0对应的静态计算图构建代码。

实践中，我们一般会先用动态计算图调试代码，然后在需要提高性能的的地方利用@tf.function切换成Autograph获得更高的效率。

### 1.3 自动微分机制

Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。

这种利用tf.GradientTape求微分的方法叫做Tensorflow的自动微分机制。

## 2. Tensorflow的层次结构

TensorFlow中有5个不同的层次结构：即硬件层，内核层，低阶API，中阶API，高阶API。

最底层为硬件层，TensorFlow支持CPU、GPU或TPU加入计算资源池。

第二层为C++实现的内核，kernel可以跨平台分布运行。

第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、计算图、自动微分. 如tf.Variable,tf.constant,tf.function,tf.GradientTape,tf.nn.softmax... 如果把模型比作一个房子，那么第三层API就是【**模型之砖**】。

第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道，特征列等等。 如tf.keras.layers,tf.keras.losses,tf.keras.metrics,tf.keras.optimizers,tf.data.DataSet,tf.feature_column... 如果把模型比作一个房子，那么第四层API就是【**模型之墙**】。

第五层为Python实现的模型成品，一般为按照OOP方式封装的高级API，主要为tf.keras.models提供的模型的类接口。 如果把模型比作一个房子，那么第五层API就是模型本身，即【**模型之屋**】。

### 2.1 低阶API（模型之砖）

低阶API主要包括张量操作，计算图和自动微分。

以线性回归模型为例，

- 构建模型

  ```python
  w = tf.Variable(tf.random.normal(w0.shape))
  b = tf.Variable(tf.zeros_like(b0,dtype = tf.float32))
  
  # 定义模型
  class LinearRegression:     
      #正向传播
      def __call__(self,x): 
          return x@w + b
  
      # 损失函数
      def loss_func(self,y_true,y_pred):  
          return tf.reduce_mean((y_true - y_pred)**2/2)
  
  model = LinearRegression()
  ```

- 准备数据及迭代器

  ```python
  #样本数量
  n = 400
  
  # 生成测试用数据集
  X = tf.random.uniform([n,2],minval=-10,maxval=10) 
  w0 = tf.constant([[2.0],[-3.0]])
  b0 = tf.constant([[3.0]])
  Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动
  
  # 构建数据管道迭代器
  def data_iter(features, labels, batch_size=8):
      num_examples = len(features)
      indices = list(range(num_examples))
      np.random.shuffle(indices)  #样本的读取顺序是随机的
      for i in range(0, num_examples, batch_size):
          indexs = indices[i: min(i + batch_size, num_examples)]
          yield tf.gather(features,indexs), tf.gather(labels,indexs)
  ```

- 训练模型

  ```python
  ##使用autograph机制转换成静态图加速
  
  @tf.function
  def train_step(model, features, labels):
      with tf.GradientTape() as tape:
          predictions = model(features)
          loss = model.loss_func(labels, predictions)
      # 反向传播求梯度
      dloss_dw,dloss_db = tape.gradient(loss,[w,b])
      # 梯度下降法更新参数
      w.assign(w - 0.001*dloss_dw)
      b.assign(b - 0.001*dloss_db)
      
      return loss
  
  def train_model(model,epochs):
      for epoch in tf.range(1,epochs+1):
          for features, labels in data_iter(X,Y,10):
              loss = train_step(model,features,labels)
          if epoch%50==0:
              printbar()
              tf.print("epoch =",epoch,"loss = ",loss)
              tf.print("w =",w)
              tf.print("b =",b)
  
  train_model(model,epochs = 200)
  ```

### 2.2 中阶API（模型之墙）

TensorFlow的中阶API主要包括各种模型层，损失函数，优化器，数据管道，特征列等等。

> 整体来说会大大的优化代码量

同样以线性回归模型为例，

- 构建模型

  ```python
  import tensorflow as tf
  from tensorflow.keras import layers,losses,metrics,optimizers
  
  model = layers.Dense(units = 1) 
  model.build(input_shape = (2,)) #用build方法创建variables
  model.loss_func = losses.mean_squared_error
  model.optimizer = optimizers.SGD(learning_rate=0.001)
  ```

- 准备数据及迭代器

  ```python
  #样本数量
  n = 400
  
  # 生成测试用数据集
  X = tf.random.uniform([n,2],minval=-10,maxval=10) 
  w0 = tf.constant([[2.0],[-3.0]])
  b0 = tf.constant([[3.0]])
  Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动
  
  #构建输入数据管道
  ds = tf.data.Dataset.from_tensor_slices((X,Y)) \
       .shuffle(buffer_size = 100).batch(10) \
       .prefetch(tf.data.experimental.AUTOTUNE)  
  ```

- 训练模型

  ```python
  #使用autograph机制转换成静态图加速
  
  @tf.function
  def train_step(model, features, labels):
      with tf.GradientTape() as tape:
          predictions = model(features)
          loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
      grads = tape.gradient(loss,model.variables)
      model.optimizer.apply_gradients(zip(grads,model.variables))
      return loss
  
  # 测试train_step效果
  features,labels = next(ds.as_numpy_iterator())
  train_step(model,features,labels)
  def train_model(model,epochs):
      for epoch in tf.range(1,epochs+1):
          loss = tf.constant(0.0)
          for features, labels in ds:
              loss = train_step(model,features,labels)
          if epoch%50==0:
              printbar()
              tf.print("epoch =",epoch,"loss = ",loss)
              tf.print("w =",model.variables[0])
              tf.print("b =",model.variables[1])
  train_model(model,epochs = 200)
  ```

### 2.3 高阶API（模型之屋）

TensorFlow的高阶API主要为tf.keras.models提供的模型的类接口。

使用Keras接口有以下**3种方式构建模型**：使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。

同样以线性回归模型为例，

- 构建模型

  ```python
  tf.keras.backend.clear_session()
  
  model = models.Sequential()
  model.add(layers.Dense(1,input_shape =(2,)))
  model.summary()
  ```

- 准备数据及迭代器

  这里不需要再构建迭代器了，可以在训练时直接指定batch_size

  ```python
  #样本数量
  n = 400
  
  # 生成测试用数据集
  X = tf.random.uniform([n,2],minval=-10,maxval=10) 
  w0 = tf.constant([[2.0],[-3.0]])
  b0 = tf.constant([[3.0]])
  Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动
  ```

- 训练模型

  ```python
  ### 使用fit方法进行训练
  
  model.compile(optimizer="adam",loss="mse",metrics=["mae"])
  model.fit(X,Y,batch_size = 10,epochs = 200)  
  
  tf.print("w = ",model.layers[0].kernel)
  tf.print("b = ",model.layers[0].bias)
  ```


## 3. 低阶API

在低阶API层次上，可以把TensorFlow当做一个增强版的numpy来使用。

TensorFlow提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。

### 3.1 张量的结构操作

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

类似numpy的一些基础操作，例如`a = tf.constant([1,2,3],dtype = tf.float32)`、`tf.eye`、`tf.zeros_like(a,dtype= tf.float32)`等

#### 3.1.1 索引切片

张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。

对于tf.Variable,可以通过索引和切片对部分元素进行修改。

对于提取张量的连续子区域，也可以使用tf.slice.

此外，对于不规则的切片提取,可以使用tf.gather,tf.gather_nd,tf.boolean_mask。

tf.boolean_mask功能最为强大，它可以实现tf.gather,tf.gather_nd的功能，并且tf.boolean_mask还可以实现布尔索引。

如果要通过修改张量的某些元素得到新的张量，可以使用tf.where，tf.scatter_nd。

#### 3.1.2 维度变换

维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose.

- tf.reshape 

  可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，所以，该操作实际上非常迅速，并且是**可逆**的。

  > 可逆意味着可以再次reshape变换为原来的张量

- tf.squeeze

  可以减少维度，如果张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度。和tf.reshape相似，它本质上不会改变张量元素的存储顺序。

- tf.expand_dims 

  可以增加维度，和tf.squeeze相反

- tf.transpose 

  可以交换维度，与tf.reshape不同，它会**改变**张量元素的存储顺序。

  tf.transpose常用于图片存储格式的变换上（# Batch,Height,Width,Channel变换为 # Channel,Height,Width,Batch）。

#### 3.1.3 合并分割

和numpy类似，可以用tf.concat和tf.stack方法对多个张量进行合并，可以用tf.split方法把一个张量分割成多个张量。

tf.concat和tf.stack有略微的区别，tf.concat是连接，不会增加维度，而tf.stack是堆叠，会增加维度。

- tf.concat

  不会增加维度，将指定的多个张量进行合并，可以指定axis=0或者axis=1。

- tf.stack

  会增加维度，将指定的多个张量进行堆叠。

- tf.split

  tf.split是tf.concat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

### 3.2 张量的数学运算

张量的操作主要包括张量的结构操作和张量的数学运算。

张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

#### 3.2.1 标量运算

标量运算符的特点是对张量实施逐元素运算。可以使用内置运算符（`+` `-` `*` `/`）和`tf.math`中函数。

#### 3.2.2 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到**一个标量**或者另外**一个向量**。 许多向量运算符都以reduce开头。

```python
#向量reduce
a = tf.range(1,10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))

out：
45
5
9
1
362880
```

```python
#张量指定维度进行reduce
b = tf.reshape(a,(3,3))
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))

out：
[[6]
 [15]
 [24]]
[[12 15 18]]
```

#### 3.2.3 矩阵运算

矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。

> 矩阵必须是二维的。类似tf.constant([1,2,3])这样的不是矩阵。

除了一些常用的运算外，大部分和矩阵有关的运算都在tf.linalg子包中。

- tf.transpose

  矩阵转置

- tf.linalg.det(a)

  矩阵行列式值

- tf.linalg.eigvals(a)

  矩阵的特征值

- tf.linalg.qr(a)

  矩阵QR分解, 将一个方阵分解为一个正交矩阵q和上三角矩阵r，QR分解实际上是对矩阵a实施Schmidt正交化得到q

- tf.linalg.svd(a)

  矩阵的svd分解

#### 3.2.4 广播机制

TensorFlow的广播规则和numpy是一样的:

- 1、如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
- 2、如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
- 3、如果两个张量在所有维度上都是相容的，它们就能使用广播。
- 4、广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
- 5、在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

`tf.broadcast_to`以显式的方式按照广播机制扩展张量的维度。

```python
#计算广播后计算结果的形状，动态形状，Tensor类型参数
c = tf.constant([1,2,3])
d = tf.constant([[1],[2],[3]])
tf.broadcast_dynamic_shape(tf.shape(c),tf.shape(d))

out：
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 3], dtype=int32)>
    
#广播效果
c+d #等价于 tf.broadcast_to(c,[3,3]) + tf.broadcast_to(d,[3,3])

out：
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]], dtype=int32)>

```

### 3.3 AutoGraph

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。Autograph机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利。

当然Autograph机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期。

#### 3.3.1 编码规范

- 函数内部尽量使用tf的函数

  被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.

  > 执行不会报错，但是某些结果不会符合预期

- 函数内部不可定义tf.Variable

  避免在@tf.function修饰的函数内部定义tf.Variable。执行会报错。

- 函数内部不可修改外部的python数据结构

  被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

  > 执行不会报错，但是某些结果不会符合预期

#### 3.3.2 机制原理

当我们使用@tf.function装饰一个函数的时候，后面发生了两件事请。一个是创建静态计算图，再一个是执行计算图。

以如下代码为例，

```python
import tensorflow as tf
import numpy as np 

@tf.function(autograph=True)
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    return c

input：
myadd(tf.constant("hello"),tf.constant("world"))

output：
tracing
0
1
2
```

相当于后面执行了

1. 创建静态计算图

   相当于在tensorflow1.0中执行了如下代码

   ```python
   g = tf.Graph()
   with g.as_default():
       a = tf.placeholder(shape=[],dtype=tf.string)
       b = tf.placeholder(shape=[],dtype=tf.string)
       cond = lambda i: i<tf.constant(3)
       def body(i):
           tf.print(i)
           return(i+1)
       loop = tf.while_loop(cond,body,loop_vars=[0])
       loop
       with tf.control_dependencies(loop):
           c = tf.strings.join([a,b])
       print("tracing")
   ```

   > 创建时计算图时，已经执行了print语句，所以字符串tracing首先被打印。

2. 执行计算图

   相当于在tensorflow1.0中执行了如下代码

   ```python
   with tf.Session(graph=g) as sess:
       sess.run(c,feed_dict={a:tf.constant("hello"),b:tf.constant("world")})
   ```

#### 3.3.3 理解编码规范

了解了以上Autograph的机制原理之后，我们也就能够理解Autograph编码规范的3条建议了。

- 函数内部尽量使用tf的函数

  解释：Python中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，普通Python函数是无法嵌入到静态计算图中的，所以 在计算图构建好之后再次调用的时候，这些Python函数并没有被计算，而TensorFlow中的函数则可以嵌入到计算图中。使用普通的Python函数会导致 被@tf.function修饰前【eager执行】和被@tf.function修饰后【静态图执行】的输出不一致。

  > 本质无法嵌入到计算图中

- 函数内部不可定义tf.Variable

  解释：如果函数内部定义了tf.Variable,那么在【eager执行】时，这种创建tf.Variable的行为在每次函数调用时候都会发生。但是在【静态图执行】时，这种创建tf.Variable的行为只会发生在第一步跟踪Python代码逻辑创建计算图时，这会导致被@tf.function修饰前【eager执行】和被@tf.function修饰后【静态图执行】的输出不一致。实际上，TensorFlow在这种情况下一般会报错。

  > tf.Variable可变，每次调用时均会执行；但是静态图执行时，只发生第一次。有可能两者不一致。

- 函数内部不可修改外部的python数据结构

  解释：静态计算图是被编译成C++代码在TensorFlow内核中执行的。Python中的列表和字典等数据结构变量是无法嵌入到计算图中，它们仅仅能够在创建计算图时被读取，在执行计算图时是无法修改Python中的列表或字典这样的数据结构变量的。

  > python的数据结果无法嵌入到静态计算图中，所以无法进行修改。

#### 3.3.4 tf.Module创建

TensorFlow提供了一个基类tf.Module，通过继承它构建子类，我们不仅可以获得以上的自然而然，而且可以非常方便地管理变量，还可以非常方便地管理它引用的其它Module，最重要的是，我们能够利用tf.saved_model保存模型并实现跨平台部署使用。

实际上，tf.keras.models.Model,tf.keras.layers.Layer 都是继承自tf.Module的，提供了方便的变量管理和所引用的子模块管理的功能。

**因此，利用tf.Module提供的封装，再结合TensoFlow丰富的低阶API，实际上我们能够基于TensorFlow开发任意机器学习模型(而非仅仅是神经网络模型)，并实现跨平台部署使用。**

使用方式有三种，

1. 继承tf.Module
2. 利用tf.Module的子类化实现封装，我们也可以通过给tf.Module添加属性的方法进行封装。
3. tf.keras中的模型和层都是继承tf.Module实现的，也具有变量管理和子模块管理功能。

## 4. 中阶API

TensorFlow的中阶API主要包括:

- 数据管道(tf.data)
- 特征列(tf.feature_column)
- 激活函数(tf.nn)
- 模型层(tf.keras.layers)
- 损失函数(tf.keras.losses)
- 评估函数(tf.keras.metrics)
- 优化器(tf.keras.optimizers)
- 回调函数(tf.keras.callbacks)

### 4.1 数据管道

如果需要训练的数据大小不大，例如不到1G，那么可以直接全部读入内存中进行训练，这样一般效率最高。

但如果需要训练的数据很大，例如超过10G，无法一次载入内存，那么通常需要在训练的过程中分批逐渐读入。

使用 tf.data API 可以构建数据输入管道，轻松处理大量的数据，不同的数据格式，以及不同的数据转换。

#### 4.1.1 数据转换为管道

可以从 Numpy array, Pandas DataFrame, Python generator, csv文件, 文本文件, 文件路径, tfrecords文件等方式构建数据管道。

其中通过Numpy array, Pandas DataFrame, 文件路径构建数据管道是最常用的方法。

通过tfrecords文件方式构建数据管道较为复杂，需要对样本构建tf.Example后压缩成字符串写到tfrecords文件，读取后再解析成tf.Example。

但tfrecords文件的优点是压缩后文件较小，便于网络传播，加载速度较快。

#### 4.1.2 管道数据变换

Dataset数据结构应用非常灵活，因为它本质上是一个Sequece序列，其每个元素可以是各种类型，例如可以是张量，列表，字典，也可以是Dataset。

Dataset包含了非常丰富的数据转换功能。

- map: 将转换函数映射到数据集每一个元素。
- flat_map: 将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。
- interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。
- filter: 过滤掉某些元素。
- zip: 将两个长度相同的Dataset横向铰合。
- concatenate: 将两个Dataset纵向连接。
- reduce: 执行归并操作。
- batch : 构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。

#### 4.1.3 提升管道性能

训练深度学习模型常常会非常耗时。

模型训练的耗时主要来自于两个部分，一部分来自**数据准备**，另一部分来自**参数迭代**。

参数迭代过程的耗时通常依赖于GPU来提升。

而数据准备过程的耗时则可以通过构建高效的数据管道进行提升。

以下是一些构建高效数据管道的建议。

- 1，使用 prefetch 方法让数据准备和参数迭代两个过程相互并行。

- 2，使用 interleave 方法可以让数据读取过程**多进程执行**,并将不同来源数据夹在一起。

- 3，使用 map 时设置num_parallel_calls 让数据转换过程多进程执行。

- 4，使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。

- 5，使用 map转换时，先batch, 然后采用向量化的转换方法对每个batch进行转换。

  > 先map相当于是多个样本同时转换，转换为batch后，相当于多个batch同时转换。

### 4.2 特征列

特征列 通常用于对结构化数据实施特征工程时候使用，图像或者文本数据一般不会用到特征列。

使用特征列可以将类别特征转换为one-hot编码特征，将连续特征构建分桶特征，以及对多个特征生成交叉特征等等。

要创建特征列，请调用 tf.feature_column 模块的函数。该模块中常用的九个函数如下图所示，所有九个函数都会返回一个 Categorical-Column 或一个 Dense-Column 对象，但却不会返回 bucketized_column，后者继承自这两个类。

注意：所有的Catogorical Column类型最终都要通过indicator_column转换成Dense Column类型才能传入模型！

- numeric_column 数值列，最常用。
- bucketized_column 分桶列，由数值列生成，可以由一个数值列出多个特征，one-hot编码。
- categorical_column_with_identity 分类标识列，one-hot编码，相当于分桶列每个桶为1个整数的情况。
- categorical_column_with_vocabulary_list 分类词汇列，one-hot编码，由list指定词典。
- categorical_column_with_vocabulary_file 分类词汇列，由文件file指定词典。
- categorical_column_with_hash_bucket 哈希列，整数或词典较大时采用。
- indicator_column 指标列，由Categorical Column生成，one-hot编码
- embedding_column 嵌入列，由Categorical Column生成，嵌入矢量分布参数需要学习。嵌入矢量维数建议取类别数量的 4 次方根。
- crossed_column 交叉列，可以由除categorical_column_with_hash_bucket的任意分类列构成。

### 4.3 激活函数

#### 4.3.1 常用激活函数

- tf.nn.sigmoid：将实数压缩到0到1之间，一般只在二分类的最后输出层使用。主要缺陷为存在梯度消失问题，计算复杂度高，输出不以0为中心。

- tf.nn.softmax：sigmoid的多分类扩展，一般只在多分类问题的最后输出层使用。

- tf.nn.tanh：将实数压缩到-1到1之间，输出期望为0。主要缺陷为存在梯度消失问题，计算复杂度高。

- tf.nn.relu：修正线性单元，最流行的激活函数。一般隐藏层使用。主要缺陷是：输出不以0为中心，输入小于0时存在梯度消失问题(死亡relu)。

- tf.nn.leaky_relu：对修正线性单元的改进，解决了死亡relu问题。

- tf.nn.elu：指数线性单元。对relu的改进，能够缓解死亡relu问题。

- tf.nn.selu：扩展型指数线性单元。在权重用tf.keras.initializers.lecun_normal初始化前提下能够对神经网络进行自归一化。不可能出现梯度爆炸或者梯度消失问题。需要和Dropout的变种AlphaDropout一起使用。

- tf.nn.swish：自门控激活函数。谷歌出品，相关研究指出用swish替代relu将获得轻微效果提升。

- gelu：高斯误差线性单元激活函数。在Transformer中表现最好。tf.nn模块尚没有实现该函数。

#### 4.3.2 模型中使用激活函数

在keras模型中使用激活函数一般有两种方式，一种是作为某些层的activation参数指定，另一种是显式添加layers.Activation激活层。

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models

tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(32,input_shape = (None,16),activation = tf.nn.relu)) #通过activation参数指定
model.add(layers.Dense(10))
model.add(layers.Activation(tf.nn.softmax))  # 显式添加layers.Activation激活层
model.summary()
```

### 4.4 **模型层**

深度学习模型一般由各种模型层组合而成。

tf.keras.layers内置了非常丰富的各种功能的模型层。例如，

layers.Dense,layers.Flatten,layers.Input,layers.DenseFeature,layers.Dropout

layers.Conv2D,layers.MaxPooling2D,layers.Conv1D

layers.Embedding,layers.GRU,layers.LSTM,layers.Bidirectional等等。

如果这些内置模型层不能够满足需求，我们也可以通过编写tf.keras.Lambda匿名模型层或继承tf.keras.layers.Layer基类构建自定义的模型层。

> 其中tf.keras.Lambda匿名模型层只适用于构造没有学习参数的模型层。

#### 4.4.1 内置模型层

**基础层**

- Dense：密集连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)
- Activation：激活函数层。一般放在Dense层后面，等价于在Dense层中指定activation。
- Dropout：随机置零层。训练期间以一定几率将输入置0，一种正则化手段。
- BatchNormalization：批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。
- SpatialDropout2D：空间随机置零层。训练期间以一定几率将整个特征图置0，一种正则化手段，有利于避免特征图之间过高的相关性。
- Input：输入层。通常使用Functional API方式构建模型时作为第一层。
- DenseFeature：特征列接入层，用于接收一个特征列列表并产生一个密集连接层。
- Flatten：压平层，用于将多维张量压成一维。
- Reshape：形状重塑层，改变输入张量的形状。
- Concatenate：拼接层，将多个张量在某个维度上拼接。
- Add：加法层。
- Subtract： 减法层。
- Maximum：取最大值层。
- Minimum：取最小值层。

**卷积网络相关层**

- Conv1D：普通一维卷积，常用于文本。参数个数 = 输入通道数×卷积核尺寸(如3)×卷积核个数
- Conv2D：普通二维卷积，常用于图像。参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数
- Conv3D：普通三维卷积，常用于视频。参数个数 = 输入通道数×卷积核尺寸(如3乘3乘3)×卷积核个数
- **SeparableConv2D**：二维深度可分离卷积层。不同于普通卷积同时对区域和通道操作，深度可分离卷积先操作区域，再操作通道。即先对每个通道做独立卷积操作区域，再用1乘1卷积跨通道组合操作通道。参数个数 = 输入通道数×卷积核尺寸 + 输入通道数×1×1×输出通道数。深度可分离卷积的参数数量一般远小于普通卷积，效果一般也更好。
- DepthwiseConv2D：二维深度卷积层。仅有SeparableConv2D前半部分操作，即只操作区域，不操作通道，一般输出通道数和输入通道数相同，但也可以通过设置depth_multiplier让输出通道为输入通道的若干倍数。输出通道数 = 输入通道数 × depth_multiplier。参数个数 = 输入通道数×卷积核尺寸× depth_multiplier。
- Conv2DTranspose：二维卷积转置层，俗称反卷积层。并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，卷积转置的输出尺寸恰好是卷积操作的输入尺寸。
- LocallyConnected2D: 二维局部连接层。类似Conv2D，唯一的差别是没有空间上的权值共享，所以其参数个数远高于二维卷积。
- MaxPool2D: 二维最大池化层。也称作下采样层。池化层无可训练参数，主要作用是降维。
- AveragePooling2D: 二维平均池化层。
- GlobalMaxPool2D: 全局最大池化层。每个通道仅保留一个值。一般从卷积层过渡到全连接层时使用，是Flatten的替代方案。
- GlobalAvgPool2D: 全局平均池化层。每个通道仅保留一个值。

**循环网络相关层**

- Embedding：嵌入层。一种比Onehot更加有效的对离散特征进行编码的方法。一般用于将输入中的单词映射为稠密向量。嵌入层的参数需要学习。
- LSTM：长短记忆循环网络层。最普遍使用的循环网络层。具有携带轨道，遗忘门，更新门，输出门。可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题。设置return_sequences = True时可以返回各个中间步骤输出，否则只返回最终输出。
- GRU：门控循环网络层。LSTM的低配版，不具有携带轨道，参数数量少于LSTM，训练速度更快。
- SimpleRNN：简单循环网络层。容易存在梯度消失，不能够适用长期依赖问题。一般较少使用。
- ConvLSTM2D：卷积长短记忆循环网络层。结构上类似LSTM，但对输入的转换操作和对状态的转换操作都是卷积运算。
- Bidirectional：双向循环网络包装器。可以将LSTM，GRU等层包装成双向循环网络。从而增强特征提取能力。
- **RNN**：RNN基本层。接受一个循环网络单元或一个循环单元列表，通过调用tf.keras.backend.rnn函数在序列上进行迭代从而转换成循环网络层。
- **LSTMCell**：LSTM单元。和LSTM在整个序列上迭代相比，它仅在序列上迭代一步。可以简单理解LSTM即RNN基本层包裹LSTMCell。
- GRUCell：GRU单元。和GRU在整个序列上迭代相比，它仅在序列上迭代一步。
- SimpleRNNCell：SimpleRNN单元。和SimpleRNN在整个序列上迭代相比，它仅在序列上迭代一步。
- AbstractRNNCell：抽象RNN单元。通过对它的子类化用户可以自定义RNN单元，再通过RNN基本层的包裹实现用户自定义循环网络层。
- Attention：Dot-product类型注意力机制层。可以用于构建注意力模型。
- AdditiveAttention：Additive类型注意力机制层。可以用于构建注意力模型。
- **TimeDistributed**：时间分布包装器。包装后可以将Dense、Conv2D等作用到每一个时间片段上。

#### 4.4.2 自定义模型层

如果自定义模型层没有需要被训练的参数，一般推荐使用Lamda层实现。

如果自定义模型层有需要被训练的参数，则可以通过对Layer基类子类化实现。

Lambda层由于没有需要被训练的参数，只需要定义正向传播逻辑即可，使用比Layer基类子类化更加简单。

Lambda层的正向逻辑可以使用Python的lambda函数来表达，也可以用def关键字定义函数来表达。

```python
import tensorflow as tf
from tensorflow.keras import layers,models,regularizers

mypower = layers.Lambda(lambda x:tf.math.pow(x,2))
mypower(tf.range(5))

out：
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 0,  1,  4,  9, 16], dtype=int32)>
```

Layer的子类化一般需要重新实现初始化方法，Build方法和Call方法。下面是一个简化的线性层的范例，类似Dense.

```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    #build方法一般定义Layer需要被训练的参数。    
    def build(self, input_shape): 
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True) #注意必须要有参数名称"w",否则会报错
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape) # 相当于设置self.built = True

    #call方法一般定义正向传播运算逻辑，__call__方法调用了它。  
    @tf.function
    def call(self, inputs): 
        return tf.matmul(inputs, self.w) + self.b
    
    #如果要让自定义的Layer通过Functional API 组合成模型时可以被保存成h5模型，需要自定义get_config方法。
    def get_config(self):  
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config
```

```python
linear = Linear(units = 16)
print(linear.built)
#如果built = False，调用__call__时会先调用build方法, 再调用call方法。
linear(tf.random.uniform((100,64))) 
print(linear.built)
config = linear.get_config()
print(config)

out：
False
True
{'name': 'linear_3', 'trainable': True, 'dtype': 'float32', 'units': 16}
```

模型保存：

```python
model.compile(optimizer = "sgd",loss = "mse",metrics=["mae"])
print(model.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))


# 保存成 h5模型
model.save("./data/linear_model.h5",save_format = "h5")
model_loaded_keras = tf.keras.models.load_model(
    "./data/linear_model.h5",custom_objects={"Linear":Linear})
print(model_loaded_keras.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))


# 保存成tf模型
model.save("./data/linear_model",save_format = "tf")
model_loaded_tf = tf.keras.models.load_model("./data/linear_model")
print(model_loaded_tf.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))
```

### 4.5 损失函数

一般来说，监督学习的目标函数由损失函数和正则化项组成。（Objective = Loss + Regularization）

对于keras模型，目标函数中的正则化项一般在各层中指定，例如使用Dense的 kernel_regularizer 和 bias_regularizer等参数指定权重使用l1或者l2正则化项，此外还可以用kernel_constraint 和 bias_constraint等参数约束权重的取值范围，这也是一种正则化手段。

损失函数在模型编译时候指定。对于回归模型，通常使用的损失函数是均方损失函数 mean_squared_error。

对于二分类模型，通常使用的是二元交叉熵损失函数 binary_crossentropy。

对于多分类模型，如果label是one-hot编码的，则使用类别交叉熵损失函数 categorical_crossentropy。如果label是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 sparse_categorical_crossentropy。

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为损失函数值。

#### 4.5.1 损失函数和正则化项

多是在层中指定。

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01), 
                activity_regularizer=regularizers.l1(0.01),
                kernel_constraint = constraints.MaxNorm(max_value=2, axis=0))) 
model.add(layers.Dense(10,
        kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))
model.compile(optimizer = "rmsprop",
        loss = "binary_crossentropy",metrics = ["AUC"])
model.summary()
```

#### 4.5.2 内置损失函数

内置的损失函数一般有类的实现和函数的实现两种形式。

如：CategoricalCrossentropy 和 categorical_crossentropy 都是类别交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。

常用的一些内置损失函数说明如下。

- mean_squared_error（均方误差损失，用于回归，简写为 mse, 类与函数实现形式分别为 MeanSquaredError 和 MSE）
- mean_absolute_error (平均绝对值误差损失，用于回归，简写为 mae, 类与函数实现形式分别为 MeanAbsoluteError 和 MAE)
- mean_absolute_percentage_error (平均百分比误差损失，用于回归，简写为 mape, 类与函数实现形式分别为 MeanAbsolutePercentageError 和 MAPE)
- Huber(Huber损失，只有类实现形式，用于回归，介于mse和mae之间，对异常值比较鲁棒，相对mse有一定的优势)
- binary_crossentropy(二元交叉熵，用于二分类，类实现形式为 BinaryCrossentropy)
- categorical_crossentropy(类别交叉熵，用于多分类，要求label为onehot编码，类实现形式为 CategoricalCrossentropy)
- sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式，类实现形式为 SparseCategoricalCrossentropy)
- hinge(合页损失函数，用于二分类，最著名的应用是作为支持向量机SVM的损失函数，类实现形式为 Hinge)
- kld(相对熵损失，也叫KL散度，常用于最大期望算法EM的损失函数，两个概率分布差异的一种信息度量。类与函数实现形式分别为 KLDivergence 或 KLD)
- cosine_similarity(余弦相似度，可用于多分类，类实现形式为 CosineSimilarity)

#### 4.5.3 自定义损失函数

**自定义损失函数接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为损失函数值。**

也可以对tf.keras.losses.Loss进行子类化，重写call方法实现损失的计算逻辑，从而得到损失函数的类的实现。

下面是一个Focal Loss的自定义实现示范。Focal Loss是一种对binary_crossentropy的改进损失函数形式。

它在样本不均衡和存在较多易分类的样本时相比binary_crossentropy具有明显的优势。

它有两个可调参数，alpha参数和gamma参数。其中alpha参数主要用于衰减负样本的权重，gamma参数主要用于衰减容易训练样本的权重。

从而让模型更加聚焦在正样本和困难样本上。这就是为什么这个损失函数叫做Focal Loss。
$$
FocalLoss(y,p) = 
\begin{cases} 
-\alpha (1-p)^{\gamma}\log(p) & \text{if y = 1} \\
-(1-\alpha) p^{\gamma}\log(1-p) & \text{if y = 0} 
\end{cases}
$$

```python
class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self,gamma=2.0,alpha=0.75,name = "focal_loss"):
        self.gamma = gamma
        self.alpha = alpha

    def call(self,y_true,y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce,axis = -1 )
        return loss
```

### 4.6 评估指标

损失函数除了作为模型训练时候的优化目标，也能够作为模型好坏的一种评价指标。但通常人们还会从其它角度评估模型的好坏。

这就是评估指标。通常损失函数都可以作为评估指标，如MAE,MSE,CategoricalCrossentropy等也是常用的评估指标。

**但评估指标不一定可以作为损失函数，例如AUC,Accuracy,Precision。因为评估指标不要求连续可导，而损失函数通常要求连续可导。**

编译模型时，可以通过列表形式指定多个评估指标。

如果有需要，也可以自定义评估指标。

自定义评估指标需要接收两个张量y_true,y_pred作为输入参数，并输出一个标量作为评估值。

也可以对tf.keras.metrics.Metric进行子类化，重写初始化方法, update_state方法, result方法实现评估指标的计算逻辑，从而得到评估指标的类的实现形式。

由于训练的过程通常是分批次训练的，而评估指标要跑完一个epoch才能够得到整体的指标结果。因此，类形式的评估指标更为常见。即需要编写初始化方法以创建与计算指标结果相关的一些中间变量，编写update_state方法在每个batch后更新相关中间变量的状态，编写result方法输出最终指标结果。

如果编写函数形式的评估指标，则只能取epoch中各个batch计算的评估指标结果的平均值作为整个epoch上的评估指标结果，这个结果通常会偏离整个epoch数据一次计算的结果。

#### 4.6.1 内置评估指标

- MeanSquaredError（均方误差，用于回归，可以简写为MSE，函数形式为mse）
- MeanAbsoluteError (平均绝对值误差，用于回归，可以简写为MAE，函数形式为mae)
- MeanAbsolutePercentageError (平均百分比误差，用于回归，可以简写为MAPE，函数形式为mape)
- RootMeanSquaredError (均方根误差，用于回归)
- Accuracy (准确率，用于分类，可以用字符串"Accuracy"表示，Accuracy=(TP+TN)/(TP+TN+FP+FN)，要求y_true和y_pred都为类别序号编码)
- Precision (精确率，用于二分类，Precision = TP/(TP+FP))
- Recall (召回率，用于二分类，Recall = TP/(TP+FN))
- TruePositives (真正例，用于二分类)
- TrueNegatives (真负例，用于二分类)
- FalsePositives (假正例，用于二分类)
- FalseNegatives (假负例，用于二分类)
- AUC(ROC曲线(TPR vs FPR)下的面积，用于二分类，直观解释为随机抽取一个正样本和一个负样本，正样本的预测值大于负样本的概率)
- CategoricalAccuracy（分类准确率，与Accuracy含义相同，要求y_true(label)为onehot编码形式）
- SparseCategoricalAccuracy (稀疏分类准确率，与Accuracy含义相同，要求y_true(label)为序号编码形式)
- MeanIoU (Intersection-Over-Union，常用于图像分割)
- TopKCategoricalAccuracy (多分类TopK准确率，要求y_true(label)为onehot编码形式)
- SparseTopKCategoricalAccuracy (稀疏多分类TopK准确率，要求y_true(label)为序号编码形式)
- Mean (平均值)
- Sum (求和)

#### 4.6.2 自定义评估指标

我们以金融风控领域常用的KS指标为例，示范自定义评估指标。

KS指标适合二分类问题，其计算方式为 KS=max(TPR-FPR).

其中TPR=TP/(TP+FN) , FPR = FP/(FP+TN)

TPR曲线实际上就是正样本的累积分布曲线(CDF)，FPR曲线实际上就是负样本的累积分布曲线(CDF)。

KS指标就是正样本和负样本累积分布曲线差值的最大值。

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,metrics

#函数形式的自定义评估指标
@tf.function
def ks(y_true,y_pred):
    y_true = tf.reshape(y_true,(-1,))
    y_pred = tf.reshape(y_pred,(-1,))
    length = tf.shape(y_true)[0]
    t = tf.math.top_k(y_pred,k = length,sorted = False)
    y_pred_sorted = tf.gather(y_pred,t.indices)
    y_true_sorted = tf.gather(y_true,t.indices)
    cum_positive_ratio = tf.truediv(
        tf.cumsum(y_true_sorted),tf.reduce_sum(y_true_sorted))
    cum_negative_ratio = tf.truediv(
        tf.cumsum(1 - y_true_sorted),tf.reduce_sum(1 - y_true_sorted))
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio)) 
    return ks_value
y_true = tf.constant([[1],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[0]])
y_pred = tf.constant([[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],[0.7],
                      [0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]])
tf.print(ks(y_true,y_pred))

out：
0.625
```

通过类实现：

继承tf.keras.metrics.Metric进行子类化，重写初始化方法, update_state方法, result方法实现评估指标的计算逻辑，从而得到评估指标的类的实现形式。

```python
#类形式的自定义评估指标
class KS(metrics.Metric):
    
    def __init__(self, name = "ks", **kwargs):
        super(KS,self).__init__(name=name,**kwargs)
        self.true_positives = self.add_weight(
            name = "tp",shape = (101,), initializer = "zeros")
        self.false_positives = self.add_weight(
            name = "fp",shape = (101,), initializer = "zeros")
   
    @tf.function
    def update_state(self,y_true,y_pred):
        y_true = tf.cast(tf.reshape(y_true,(-1,)),tf.bool)
        y_pred = tf.cast(100*tf.reshape(y_pred,(-1,)),tf.int32)
        
        for i in tf.range(0,tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(
                    self.true_positives[y_pred[i]]+1.0)
            else:
                self.false_positives[y_pred[i]].assign(
                    self.false_positives[y_pred[i]]+1.0)
        return (self.true_positives,self.false_positives)
    
    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(
            tf.cumsum(self.true_positives),tf.reduce_sum(self.true_positives))
        cum_negative_ratio = tf.truediv(
            tf.cumsum(self.false_positives),tf.reduce_sum(self.false_positives))
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio)) 
        return ks_value
y_true = tf.constant([[1],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[0]])
y_pred = tf.constant([[0.6],[0.1],[0.4],[0.5],[0.7],[0.7],
                      [0.7],[0.4],[0.4],[0.5],[0.8],[0.3],[0.5],[0.3]])

myks = KS()
myks.update_state(y_true,y_pred)
tf.print(myks.result())

out：
0.625
```

### 4.7 优化器

机器学习界有一群炼丹师，他们每天的日常是：

**拿来药材（数据），架起八卦炉（模型），点着六味真火（优化算法），就摇着蒲扇等着丹药出炉了。**

**不过，当过厨子的都知道，同样的食材，同样的菜谱，但火候不一样了，这出来的口味可是千差万别。火小了夹生，火大了易糊，火不匀则半生半糊。**

机器学习也是一样，模型优化算法的选择直接关系到最终模型的性能。有时候效果不好，未必是特征的问题或者模型设计的问题，很可能就是优化算法的问题。

深度学习优化算法大概经历了 SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam 这样的发展历程。

对于一般新手炼丹师，优化器直接使用Adam，并使用其默认参数就OK了。

一些爱写论文的炼丹师由于追求评估指标效果，可能会偏爱前期使用Adam优化器快速下降，后期使用SGD并精调优化器参数得到更好的结果。

此外目前也有一些前沿的优化算法，据称效果比Adam更好，例如LazyAdam, Look-ahead, RAdam, Ranger等.

#### 4.7.1 优化器的使用

**优化器主要使用apply_gradients方法传入变量和对应梯度从而来对给定变量进行迭代，或者直接使用minimize方法对目标函数进行迭代优化。**

当然，更常见的使用是在编译时将优化器传入keras的Model,通过调用model.fit实现对Loss的的迭代优化。

初始化优化器时会创建一个变量optimier.iterations用于记录迭代的次数。因此优化器和tf.Variable一样，一般需要在@tf.function外创建。

#### 4.7.2 内置优化器

深度学习优化算法大概经历了 SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam 这样的发展历程。

在keras.optimizers子模块中，它们基本上都有对应的类的实现。

- SGD, 默认参数为纯SGD, 设置momentum参数不为0实际上变成SGDM, 考虑了一阶动量, 设置 nesterov为True后变成NAG，即 Nesterov Accelerated Gradient，在计算梯度时计算的是向前走一步所在位置的梯度。
- Adagrad, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率。缺点是学习率单调下降，可能后期学习速率过慢乃至提前停止学习。
- RMSprop, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对Adagrad进行了优化，通过指数平滑只考虑一定窗口内的二阶动量。
- Adadelta, 考虑了二阶动量，与RMSprop类似，但是更加复杂一些，自适应性更强。
- Adam, 同时考虑了一阶动量和二阶动量，可以看成RMSprop上进一步考虑了一阶动量。
- Nadam, 在Adam基础上进一步考虑了 Nesterov Acceleration。

### 4.8 回调函数

tf.keras的回调函数实际上是一个类，一般是在model.fit时作为参数指定，用于控制在训练过程开始或者在训练过程结束，在每个epoch训练开始或者训练结束，在每个batch训练开始或者训练结束时执行一些操作，例如收集一些日志信息，改变学习率等超参数，提前终止训练过程等等。

同样地，针对model.evaluate或者model.predict也可以指定callbacks参数，用于控制在评估或预测开始或者结束时，在每个batch开始或者结束时执行一些操作，但这种用法相对少见。

大部分时候，keras.callbacks子模块中定义的回调函数类已经足够使用了，如果有特定的需要，我们也可以通过对keras.callbacks.Callbacks实施子类化构造自定义的回调函数。

所有回调函数都继承至 keras.callbacks.Callbacks基类，拥有params和model这两个属性。

其中params 是一个dict，记录了训练相关参数 (例如 verbosity, batch size, number of epochs 等等)。

model即当前关联的模型的引用。

此外，对于回调类中的一些方法如on_epoch_begin,on_batch_end，还会有一个输入参数logs, 提供有关当前epoch或者batch的一些信息，并能够记录计算结果，如果model.fit指定了多个回调函数类，这些logs变量将在这些回调函数类的同名函数间依顺序传递。

#### 4.8.1 内置回调函数

- BaseLogger： 收集每个epoch上metrics在各个batch上的平均值，对stateful_metrics参数中的带中间状态的指标直接拿最终值无需对各个batch平均，指标均值结果将添加到logs变量中。该回调函数被所有模型默认添加，且是第一个被添加的。
- History： 将BaseLogger计算的各个epoch的metrics结果记录到history这个dict变量中，并作为model.fit的返回值。该回调函数被所有模型默认添加，在BaseLogger之后被添加。
- EarlyStopping： 当被监控指标在设定的若干个epoch后没有提升，则提前终止训练。
- TensorBoard： 为Tensorboard可视化保存日志信息。支持评估指标，计算图，模型参数等的可视化。
- ModelCheckpoint： 在每个epoch后保存模型。
- ReduceLROnPlateau：如果监控指标在设定的若干个epoch后没有提升，则以一定的因子减少学习率。
- TerminateOnNaN：如果遇到loss为NaN，提前终止训练。
- LearningRateScheduler：学习率控制器。给定学习率lr和epoch的函数关系，根据该函数关系在每个epoch前调整学习率。
- CSVLogger：将每个epoch后的logs结果记录到CSV文件中。
- ProgbarLogger：将每个epoch后的logs结果打印到标准输出流中。

#### 4.8.2  自定义回调函数

可以使用`callbacks.LambdaCallback`编写较为简单的回调函数，也可以通过对`callbacks.Callback`子类化编写更加复杂的回调函数逻辑。

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,metrics,callbacks
import tensorflow.keras.backend as K 

# 示范使用LambdaCallback编写较为简单的回调函数

import json
json_log = open('./data/keras_log.json', mode='wt', buffering=1)
json_logging_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps(dict(epoch = epoch,**logs)) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# 示范通过Callback子类化编写回调函数（LearningRateScheduler的源代码）

class LearningRateScheduler(callbacks.Callback):
    
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                 'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
```

## 5. 高阶API

TensorFlow的高阶API主要是`tensorflow.keras.models`.

本章我们主要详细介绍tensorflow.keras.models相关的以下内容。

- 模型的构建（Sequential、functional API、Model子类化）
- 模型的训练（内置fit方法、内置train_on_batch方法、自定义训练循环、单GPU训练模型、多GPU训练模型、TPU训练模型）
- 模型的部署（tensorflow serving部署模型、使用spark(scala)调用tensorflow模型）

### 5.1 构建模型3种方法

#### 5.1.1 sequential方式

```python
tf.keras.backend.clear_session()

model = models.Sequential()

model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation = "sigmoid"))

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()
```

#### 5.1.2 函数API方式

```python
tf.keras.backend.clear_session()

inputs = layers.Input(shape=[MAX_LEN])
x  = layers.Embedding(MAX_WORDS,7)(inputs)

branch1 = layers.SeparableConv1D(64,3,activation="relu")(x)
branch1 = layers.MaxPool1D(3)(branch1)
branch1 = layers.SeparableConv1D(32,3,activation="relu")(branch1)
branch1 = layers.GlobalMaxPool1D()(branch1)

branch2 = layers.SeparableConv1D(64,5,activation="relu")(x)
branch2 = layers.MaxPool1D(5)(branch2)
branch2 = layers.SeparableConv1D(32,5,activation="relu")(branch2)
branch2 = layers.GlobalMaxPool1D()(branch2)

branch3 = layers.SeparableConv1D(64,7,activation="relu")(x)
branch3 = layers.MaxPool1D(7)(branch3)
branch3 = layers.SeparableConv1D(32,7,activation="relu")(branch3)
branch3 = layers.GlobalMaxPool1D()(branch3)

concat = layers.Concatenate()([branch1,branch2,branch3])
outputs = layers.Dense(1,activation = "sigmoid")(concat)

model = models.Model(inputs = inputs,outputs = outputs)

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()
```

#### 5.1.3 Model子类化

```python
# 先自定义一个残差模块，为自定义Layer

class ResBlock(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self,input_shape):
        self.conv1 = layers.Conv1D(filters=64,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv2 = layers.Conv1D(filters=32,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv3 = layers.Conv1D(filters=input_shape[-1],
                                   kernel_size=self.kernel_size,activation = "relu",padding="same")
        self.maxpool = layers.MaxPool1D(2)
        super(ResBlock,self).build(input_shape) # 相当于设置self.built = True
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.Add()([inputs,x])
        x = self.maxpool(x)
        return x
    
    #如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):  
        config = super(ResBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
```



```python
# 自定义模型，实际上也可以使用Sequential或者FunctionalAPI

class ImdbModel(models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()
        
    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS,7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense = layers.Dense(1,activation = "sigmoid")
        super(ImdbModel,self).build(input_shape)
    
    def call(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = layers.Flatten()(x)
        x = self.dense(x)
        return(x)
```

### 5.2 训练模型3种方法

模型的训练主要有内置fit方法、内置tran_on_batch方法、自定义训练循环。

> fit_generator方法在tf.keras中不推荐使用，其功能已经被fit包含。

#### 5.2.1 内置fit方法

该方法功能非常强大, 支持对numpy array, tf.data.Dataset以及 Python generator数据进行训练。

并且可以通过设置回调函数实现对训练过程的复杂控制逻辑。

#### 5.2.2 内置train_on_batch方法

该内置方法相比较fit方法更加灵活，可以不通过回调函数而直接在批次层次上更加精细地控制训练的过程。

```python
def train_model(model,ds_train,ds_valid,epoches):

    for epoch in tf.range(1,epoches+1):
        model.reset_metrics()
        
        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")
        
        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y,reset_metrics=False)
            
        if epoch%1 ==0:
            printbar()
            tf.print("epoch = ",epoch)
            print("train:",dict(zip(model.metrics_names,train_result)))
            print("valid:",dict(zip(model.metrics_names,valid_result)))
            print("")
```

#### 5.2.3 自定义训练循环

自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性。

### 5.3 单GPU训练

无论是内置fit方法，还是自定义训练循环，从CPU切换成单GPU训练模型都是非常方便的，**无需更改任何代码**。当存在可用的GPU时，如果不特意指定device，tensorflow会自动优先选择使用GPU来创建张量和执行张量计算。

设置GPU

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
    tf.config.set_visible_devices([gpu0],"GPU") 
```

### 5.4 多GPU训练

MirroredStrategy过程简介：

- 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
- 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
- N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
- 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
- 使用梯度求和的结果更新本地变量（镜像变量）；
- 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

```python
#此处在colab上使用1个GPU模拟出两个逻辑GPU进行多GPU训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑GPU模拟多GPU训练
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
```

训练时需要额外加入两行代码

```python
#增加以下两行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)
    
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)  
```

### 5.5 TPU训练

如果想尝试使用Google Colab上的TPU来训练模型，也是非常方便，仅需添加6行代码。

训练时需要额外加入六行代码

```python
#增加以下6行代码
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    model.summary()
    model = compile_model(model)
```

### 5.6 模型部署

TensorFlow训练好的模型以tensorflow原生方式保存成protobuf文件后可以用许多方式部署运行。

例如：通过 tensorflow-js 可以用javascrip脚本加载模型并在浏览器中运行模型。

通过 tensorflow-lite 可以在移动和嵌入式设备上加载并运行TensorFlow模型。

通过 tensorflow-serving 可以加载模型后提供网络接口API服务，通过任意编程语言发送网络请求都可以获取模型预测结果。

通过 tensorFlow for Java接口，可以在Java或者spark(scala)中调用tensorflow模型进行预测。

我们主要介绍tensorflow serving部署模型、使用spark(scala)调用tensorflow模型的方法。

### 5.7 spark-scala调用tf model

如果使用pyspark的话会比较简单，只需要在每个executor上用Python加载模型分别预测就可以了。

但工程上为了性能考虑，通常使用的是scala版本的spark。



spark-scala调用tensorflow模型概述

在spark(scala)中调用tensorflow模型进行预测需要完成以下几个步骤。

（1）准备protobuf模型文件

（2）创建spark(scala)项目，在项目中添加java版本的tensorflow对应的jar包依赖

（3）在spark(scala)项目中driver端加载tensorflow模型调试成功

（4）在spark(scala)项目中通过RDD在executor上加载tensorflow模型调试成功

（5） 在spark(scala)项目中通过DataFrame在executor上加载tensorflow模型调试成功