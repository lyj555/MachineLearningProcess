[TOC]

# Model Process



## 1. 定义预测问题

## 2. 确定预测问题的衡量指标

## 3. 确认模型的评估方式

## 4. 准备数据

## 5. 开发模型

下面分两部分介绍，分别是**机器学习**模型和**深度学习**模型的创建流程。

### 5.1 机器学习

对于二维的数据类型（样本，特征），往往会采用机器学习模型进行建模。

目前最常用的模型主要为逻辑回归模型、bagging和boosting类模型。

大致流程分为**数据探查及处理，特征工程，特征选择，超参数的选择，模型评估，模型融合**

#### 5.1.1 数据探查及处理

一般此时拿到的数据有两种情形，一种是原始日志数据，一种是已经聚合好的数据。

- 原始日志数据

  这种数据，一个主键往往包含多条记录，之后需要基于此主键进行聚合（生成特征）。

- 已经聚合的数据

  此时数据，一个主键会对应一条记录，当然也可以基于这份数据再衍生新的特征。

> 多数情况下，此时拿到的是已经聚合好的数据，这一步往往通过hive或者spark来完成

清楚自己所有的数据，查看数据行数和列数，看一下每一列的类型，唯一值，缺失值个数等等属性信息，确保数据的格式和内容准确且不要发生**标签泄露**（个人觉得这一步虽然简单，但是确及其重要，如果数据存在问题，将会直接影响到后面整个流程）。

在清楚自己数据后，可以做一些数据清洗工作，比如剔除一些不相关字段，删除一些缺失值过多的属性；如果发现数据存在不均衡的情况，可以对正负样本做采样。

整体来看，这一步就是要确认数据无误，然后在此基础上做些基本的清洗工作。

#### 5.1.2 特征工程

在机器学习领域中，**特征工程是最核心的部分**。

个人理解，往往特征分为两部分，一部分是**业务特征**，另一部分是**变换特征**。

##### 5.1.2.1 业务特征

这部分特征大致分为三个方向，

- 基本属性特征

  如果训练主体是人，比如年龄、性别、身高、体重等都可以算作基础属性特征；往往把这方面的特征分为**空间特征**（种类、数量、大小、长度等）和**时间特征**（时长、次数、频率、周期等）。

- 统计特征

  这部分特征是多数人最容易想到的。比如说某个人花费金额的平均值、中位数、最大值、最小值、中位数、标准差和偏度峰度等等

- 复杂特征

  对主体进行更细维度的刻画。往往是上面两种特征的组合。

  - 时间特征*空间特征

    如最近三个月的购物次数

  - 空间特征*空间特征

    超过500元的订单数量

  - 时间\*空间\*统计

    最早的三个月的购物次数占总购物次数的比重

  可以按照这种思路添加其他业务维度特征，来进行交叉。

> 以上谈及特征提取指可以度量的特征，但是像图像、文本和语音这种数据不太适用此种方式，此时可以通过深度网络的方式进行特征的提取，这部分数据会选择深度学习模型处理（可以自动进行特征提取）。

##### 5.1.2.2 变换特征

这部分特征往往是基于现有特征做了一些数学上的变换。主要参考[sklearn.preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)

- 连续型特征

  - 标准化  
    $x^{\prime}=\frac{x-\bar{x}}{S}$
  
  - 归一化
  
    $x^{\prime}=\frac{x-Min}{Max-Min}$
    
    > 标准化强调经数据放缩到同一个量纲，而归一化只是做了一个线性变换，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”
    
  - 离散化

    按照特征的不同取值将特征离散化为k个不同的bins。

  - 异常值处理

    可以按照IQR（Q3-Q1）的方式，[Q3-1.5IQR, Q3+1.5IQR]范围限定为正常特征范围，超出此范围认定为异常值。

  - 非线性变化

    往往数据存在长尾分布时，采用log变换，将数据变换至正态分布。

    另外在sklearn.preprocessing中提到了两种非线性变换方法，quantile transforms和power transforms ，这两种变换均保留了原始数据中存在的大小关系。

    - quantile transforms

      it provide a non-parametric transformation to map the data to a **uniform distribution** with values between 0 and 1

    - power transforms 

      Power transforms are a family of parametric, monotonic transformations that aim to map data from any distribution to as close to a **Gaussian distribution** as possible in order to stabilize variance and minimize skewness

    > 在非线性变换这部分中，个人经常看到的是做log变换，即将长尾分布转换为正态分布。
  
  - 多项式特征
  
    如原有特征为$feature_1, feature_2$，生成的多项式特征可以为$feature_1^2, feature_2^2, feature_1*feature_2$。从这种形式可以看出，这种做法的可解释性较差，往往在一些竞赛中，为了更高成绩，来通过这种方式来衍生新特征。
  
- 离散型特征

  往往有两种做法，一个是one-hot，另外一个直接按照index编码（仍然是一列，不过变为0,1,2,3...的数连续型特征）。
  
  假设某个离散型特征F取值为三个，分别为A,B,C，那么经过one-hot后，会生成三个特征，可以将其命名为is_A, is_B, is_C，分别表示特征F取值是否等于类别A，是否等于类别B和是否等于类别C。而按照index编码后，只生成一列特征，将其命名为f_index，其取值为0, 1, 2，相当于将做了一个映射{A: 0, B: 1, C: 2}
  
  个人理解，在离散型特征取值个数10个内时，可以采用one-hot方式，如果大于10个，则可以将其index编码，虽然index这种编码方式欠妥，但是大量经验来看，这两种方式的效果接近，但是按照index编码的方式只会生成一列特征，而one-hot会衍生多列特征，极大增加了训练的时间。

##### 5.1.2.3 特征提取思路

特征工程的初始阶段，往往首先做业务方面的特征，个人理解，应当遵循下述大致步骤，

1. 根据业务理解或者经验快速构建业务特征（不要想着一步就把所有的特征全做出来）
2. 根据这部分特征来构建一个模型，通过模型来反馈所构建特征的重要性（特征重要性是否符合自己的预期、有哪些意料之外的特征比较重要，为什么）
3. 通过这种反馈来指引你下一步特征工程优化的方向

*在不断的追问和查看数据的过程中，慢慢的积累领域知识和常识，形成自己的经验，形成对目标客群的深入认识。*

在业务特征过程大致完结后，可以加入一些变换特征，个人经验，在变换特征中，经常使用的是连续类型的离散化和非线性变换以及离散特征的index编码，可以看出核心的特征均在业务特征方面。

>目前特征工程，已经有一些比较火热的特征生成的包，比如[featuretools](https://www.featuretools.com/)和[tsfresh](https://tsfresh.readthedocs.io/en/latest/)
>
>- featuretools
>
>  多个主键的聚合特征，如果原始日志数据中存在多个主体，那么他可以先基于一个主体进行聚合然后在最终训练的主体再次聚合，这种方式的聚合往往容易忽略，所以个人觉得featuretools生成的一些特征可以作为补充加入至自己的特征中。
>
>- tsfresh
>
>  主要针对时序的序列提取特征，可以说它将所有时序相关的特征均考虑进去，从而生成大量的特征，同样可以作为补充加入至自己的特征中。

#### 5.1.3 特征选择

这部分目前主要有三类方法，分别为**过滤式特征选择、包裹式特征选择和嵌入式特征选择**。

- 过滤式特征选择

  相当于单变量特征选择，每个特征列和标签列计算一个公式来衡量该特征和标签的相关性。

  - 分类任务

    - 信息增益

      $f(A, D)=H(D)-H(D|A)$

    - 基尼系数

      $f(A,D)=gini(D, A)$

    - 卡方检验

      [`chi2`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)

    - 方差分析

      [`f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)（ANOVA F-value between label/feature）

    - 互信息

      [`mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)

  - 回归任务

    - 相关系数

      皮尔逊相关系数

    - 方差分析

      [`f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

    - 互信息

      [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression)

  > 上面提及的过滤式特征选择主要是标签相关的，另外还有单纯的特征筛选（不依赖标签列），比如常用的两个是按照列缺失比例和列标准差进行筛选（往往需要删除缺失率较大的列和标准差较小列）。

- 包裹式特征选择

  这种方式直接把最终使用的学习器的性能作为特征子集的评价准则，整体效果要优于过滤式特征选择，但是计算开销通常比过滤式特征选择要大得多。

  - LVW(Las Vegas Wrapper)

    这是一个典型的包裹式特征选择方法，每次通过随机选择的方式生成特征子集，然后通过学习器进行评估，然后迭代此过程直至达到设定的条件，最终返回相对最优的特征子集。

  > LVW的采用随机的策略生成特征子集，显然我们可以根据自己业务理解或者其它方式生成我们自己定制的特征子集

- 嵌入式特征选择

  以上两种特征选择方式，可以这样理解，过滤式特征选择相当于是在模型训练之前进行特征选择，包裹式特征选择则是在模型训练之后进行特征选择。而嵌入式特征选择恰是在模型训练的同时进行了特征选择。

  通常这种方式是通过在损失函数中加入正则项来实现，通常是加入L1正则项，类似决策树相关的模型，其实也可以看做是嵌入式特征选择的一种，其在构建树的同时就进行了特征选择。

#### 5.1.4 超参数选择

目前超参数的选择应该是有四种方式（个人总结），分别为网格搜索、随机搜素、BayesianOptimization和Hyper-parameter Optimizationt方式。

- 网格搜索

  sklearn的实现方式[`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)，但是输入的数据必须要做K折交叉验证，往往比较耗时。

- 随机搜索

  slearn的实现方式[`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)，类似sklearn的网格搜索，需要做交叉验证，但是不需要遍历所有的网格参数，这样很大程度可以加快超参数的选择时间。

- BayesianOptimization

  [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)构建需要优化的函数的先验分布（高斯过程），随着迭代论述的增加，先验分布发生改变，算法逐步缩小需要优化的参数空间，从而找到最优参数集。

- Hyper-parameter Optimization

  [Hyper-parameter Optimization](https://github.com/hyperopt/hyperopt) is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

#### 5.1.5 模型选择



#### 5.1.6 模型融合




#### 5.1.7 案例



### 5.2 深度学习



## 6. 模型的解释

## 7. 模型线上部署以及测试

## 8. 效果监控

## 9. 模型的迭代



## References

- [如何提取特征？（第一版）](https://zhuanlan.zhihu.com/p/25743799)