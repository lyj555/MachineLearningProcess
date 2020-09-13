[TOC]

# Model Process

## 1. 确认模型任务

1. 定义模型预测问题

   往往来说业务侧的需求是变化多样的，所以需要算法人员及时将这种业务侧的需求转化为模型的预测问题。这一步可以说是整个模型构建的核心，如若模型预测问题定义不清，后面的所有工作可能都是白费力气。

   这一步核心有两点，

   - 是否问题具有可预测性

     首次要从业务的角度以及数据的角度明确这个问题是否具有预测性，比如要预测股票未来30天内每天的涨幅，显然这是一个不可预测的问题，因为未来的涨幅影响的因素太多，而且未来的信息变化不能获取到。

   - 预测目标是如何定义

     确认预测的目标的详细定义。

     比如，业务方想找出流失可能性大的用户，此时需要我们定义清楚流失用户的定义，如前n天内有行为，接下来m天内无任何行为，这里需要我们定义清楚三个点，n是多少，行为具体指哪些行为以及m是多少。

2. 明确输入和输出数据

   在定义清楚预测问题后，接下来需要明确输入数据是什么，预测标签是什么，只有拥有可用的训练数据，才能对标签进行预测。

   - 明确输入数据

     输入数据，也即我们的训练数据，查看是否有足够的训练数据，以及数据的质量。

     如果可用的训练的数据可能就有几百条，或者数据量很大但是缺失率极高，那么即使问题可预测，但是训练数据不足以进行训练。

   - 明确输出数据

     输出数据，也即我们的标签数据，第一步中我们已经定义清楚模型的预测问题，接下来需要我们明确我们 的输出数据。

     这里要明确两个问题，

     - 能否获取标签数据

       按照定义的方式确认能否获取到标签数据，比如说我们要提取文本中的业务类的实体信息，虽然定义及其明确，但是我们不具有此类数据标签，此时我们需要协调资源看看是否需要标注数据。

     - 标签分布是否正常

       按照第一步的定义，查看一下我们标签数据的分布是否正常，根据结论决定是否调整预测目标的定义（需要注意并非第一步的定义是绝对正确）

## 2. 确认模型的评估方式

这里的评估方式主要分为两个大部分，一个是**业务层面**的评估指标，这个取决于具体的业务；另外一个则是**模型层面**的指标，用于衡量模型训练的效果，下面主要列举一下模型层面的评估方式。

### 2.1 分类模型

#### 2.1.1 二分类

- 混淆矩阵

  |          |      真实为正      |      真实为负      |
  | :------: | :----------------: | :----------------: |
  | 预测为正 | TP(True Positive)  | FP(False Negative) |
  | 预测为负 | FN(False Negative) | TN(True Negative)  |

  > 理解方式：
  >
  > 首字母分为T和F，表示True和False，也就是表示**预测的对和错**；第二个字母分为P和F，表示Positive和Negative，表示**预测正例和负例**。（核心是要理解首位置字母和第二位置字母表达的意思）
  >
  > 对于一个样本来说，如果它是TP，表示模型预测它为正例且预测正确，说明这个样本真实即为正例且模型预测正确；如果它是FP，表示模型预测它为正例但预测错误，说明这个样本真是为负例但模型预测错误。

- 精准率&召回率&准确率

  - 精准率

    预测为正例的样本样本中，真实也为正例的样本占的比率（字面意思就是精&准）

    计算方式：$precision = \frac{TP}{TP+FP}$

  - 召回率

    所有真实的正例样本中，预测为正例的样本占的比率（字面意思即为召回了多少真实样本）

    计算方式：$recall=\frac{TP}{TP+FN}$

  - 准确率

    预测正确的样本占总样本的比率（最容易理解）

    计算方式：$accuracy=\frac{TP+TN}{TP+TN+FP+FN}$

- $F1$&$F_\beta$

  如果想综合查看精准率和召回率，此时用$F1$和$F_\beta$。

  - F1

    精准率和召回率的调和平均，$F1=\frac{1}{2}(\frac{1}{precision}+\frac{1}{recall})$。

  - $F_\beta$

    加权的精准率和召回率，$F_\beta=\frac{1}{1+\beta^2}(\frac{1}{precision}+\frac{\beta^2}{recall})\quad(\beta>0)$。

    其中$\beta$度量了召回率对于精确率的相对重要性。可以看出，如果$\beta=1$，$F_\beta$退化为F1，如果$\beta>1$，召回率占有更大比重；如果$\beta<1$，精准率占有更大的比重。

- AUC(Area Under Curve)

  它衡量的整体样本的排序能力。 将测试样本通过预测为正例的概率值由高到低进行排序，“最可能”的正例排在最前面，“最不可能”的正例排在最后面，整个样本的分类相当于取每个样本的预测概率值为分割点，然后计算FPR和TPR，这样最终得到n组的FPR和TPR，然后横坐标为FPR，纵坐标为TPR得到ROC曲线，AUC即为ROC曲线下的面积。

  其中TPR表示真正例率，计算公式$TPR=\frac{TP}{TP+FN}$。可以看出这个和召回率公式是一样的；

  其中FPR表示假正例率，计算公式$FPR=\frac{FP}{FP+TN}$。

  > TPR和FPR都是针对预测为正例的样本而言；TPR的分母是所有的真实正例样本，而FPR的分母则为所有的真实负例样本；显然TPR越大越好，而FPR则越小越好。

  AUC的本质（物理含义）：

  **随机挑一个正样本和负样本，正样本的得分大于负样本得分的可能性。**

  > 相当于是auc衡量了模型对正负样本的排序能力，auc越接近于1，说明模型对正样本的识别能力（相比较于负样本）准确。

  所以其计算方式还有另一种：
  
  假设总共有（m+n）个样本，其中正样本m个，负样本n个，总共有m\*n个正负样本对，计数，正样本预测为正样本的概率值大于负样本预测为正样本的概率值记为1，累加计数，然后除以（m*n）就是AUC的值

#### 2.1.2 多分类

同样对应于recall和precision以及auc。

对应的方式有`micro`，`macro`, `weighted`和`sample`

### 2.2 回归模型

常用的四种，分别为MSE、RMSE、MAE和R Squared。

- MSE(Mean Squared Error)

  $\frac{1}{m}\sum_{i=1}^m (y_i - \hat{y_i})^2$

- RMSE(Root Mean Squared Error)

  $\sqrt{\frac{1}{m}\sum_{i=1}^m (y_i - \hat{y_i})^2}$

  > 相比较于MSE，RMSE将误差的量级化到数据一个级别

- MAE(Mean Absolute Error)

  $\frac{1}{m} \sum_{i=1}^m |y_i - \hat{y_i}|$

- R Squared

  $1-\frac{\sum_{i=1}^m (y_i - \hat{y_i})^2}{\sum_{i=1}^m (y_i - \bar{y_i})^2}$
  
  R方的含义是，预测值解释了 $y_i$ 变量的方差的多大比例，衡量的是预测值对于真值的拟合好坏程度。通俗理解，假定$y_i$ 的方差为1个单位，则R方表示"使用该模型之后，  $y_i$的残差的方差减少了多少"。比如R方等于0.8，则使用该模型之后残差的方差为原始 $y_i$ 值方差的20%。
  
  - R方=1：最理想情况，所有的预测值等于真值。
  
  - R方=0：一种可能情况是"简单预测所有y值等于y平均值"，即所有 $\hat{y_i}$ 都等于$\bar{y}$（即真实y值的平均数），但也有其他可能。
  
  - R方<0：模型预测能力差，比"简单预测所有y值等于y平均值"的效果还差。这表示可能用了错误模型，或者模型假设不合理。
  
  - R方的最小值没有下限，因为预测可以任意程度的差。因此，R方的范围是$(-\infty, 1]$ 。
  
  > 注意：R方并不是某个数的平方，因此可以是负值。

### 2.3 排序模型

主要有`MAP(Mean Average Precision)`、`nDCG(Normalized Discounted Cumulative Gain)`、AUC

参考博客[推荐系统排序（Ranking）评价指标](https://www.cnblogs.com/shenxiaolin/p/9309749.html)、[NDCG](https://www.jianshu.com/p/51aca7559218)

## 3. 准备数据

## 4. 开发模型

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

  > 上面提及的过滤式特征选择主要是标签相关的，另外还有单纯的特征筛选（不依赖标签列），比如常用的两个是按照**列缺失比例**和**列标准差**进行筛选（往往需要删除缺失率较大的列和标准差较小列）。

- 包裹式特征选择（已代码实现）

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

  [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)构建需要优化的函数的先验分布（高斯过程），随着迭代论述的增加，先验分布发生改变，算法逐步缩小需要优化的参数空间，从而找到最优参数集。可参考本项目当前目录下`ParamBayesOptimization.md`

- Hyper-parameter Optimization

  [Hyper-parameter Optimization](https://github.com/hyperopt/hyperopt) is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

> 网格搜索和随机搜索以及基于spark的分布式版本均已代码实现，BayesianOptimization和Hyper-parameter Optimization基于其开源包结合自己的逻辑进行了封装，也已经代码实现。

#### 5.1.5 模型选择



#### 5.1.6 模型融合




#### 5.1.7 案例



### 5.2 深度学习

#### 5.2.1 数据探查及处理

##### 5.2.1.1 NLP

##### 5.2.1.2 图像

#### 5.2.2 模型选择

#### 5.2.3 超参数选择

#### 5.2.4 案例

##### 5.2.4.1 文本分类

##### 5.2.4.2 图像分类

##### 5.2.4.3 机器翻译



## 6. 模型的解释

### 6.1 树模型

### 6.2 深度模型



## 7. 模型线上部署以及测试

## 8. 效果监控

## 9. 模型的迭代



## References

- [如何提取特征？（第一版）](https://zhuanlan.zhihu.com/p/25743799)
- [回归评价指标](https://www.jianshu.com/p/9ee85fdad150)
- [R方的理解与用法](https://zhuanlan.zhihu.com/p/143132259)