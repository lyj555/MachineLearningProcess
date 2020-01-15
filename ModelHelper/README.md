# Model Helper

The purpose of the project is to make modelers more efficient.

目录
--

- 1._Installation
- 2._特征选择

    - 2.1 random_search（随机搜索）
    - 2.2 lvw（Las Vegas Wrapper）
    - 2.3 weight search(加权选择)
    - 2.4 top feature importance
    - 2.5 分布式特征选择
      - 2.5.1 random search
      - 2.5.2 weight search
      - 2.5.3 top feature importance
- 3._超参数选择
    - 3.1 随机选择
    
    - 3.2 网格搜索
    
    - 3.3 bayes_opt
    
    - 3.4 hyper_opt
- 4._模型解释
  
    - 4.1 机器学习模型
        - 4.1.1 特征选取策略
          - 4.1.1.1 最大shap值
          - 4.1.1.2 最小shap值
          - 4.1.1.3 shap绝对值
          - 4.1.1.4 概率阈值
        - 4.1.2 特征分组
        - 4.1.3 自定义输出的格式
    
    - 4.2 深度模型解释
        - 4.2.1 图像 
          - 4.2.1.1 构建模型及训练
          - 4.2.1.2 解释预测样本
        - 4.2.2 文本
          - 4.2.2.1 构建模型及训练
          - 4.2.2.2 解释预测样本
- 5._模型融合
    - 5.1 Stacking
      - 5.1.1 单机版本
      - 5.1.2 分布式版本
    - 5.2 Blending
      - 5.2.1 单机版本
      - 5.2.2 分布式版本
    - 5.3 Bagging
      - 5.3.1 单击版本
      - 5.3.2 分布式版本
- TO DO LIST




## 1. Installation

- 直接使用

    直接复制文件夹`model_helper`至本地项目， 然后import使用即可

- 本地安装

    - `git clone http://igit.58corp.com/AIgroups/ModelHelper.git` 

    - `python setup.py install`

- pip安装

    `pip install git+http://igit.58corp.com/AIgroups/ModelHelper.git`


## 2. 特征选择

目前特征选择部分实现了四种算法，分别为**随机搜索**、**lvw**、**random search by feature importance**和**top feature importance**，这几种算法本质区别主要是每次特征子集的选取不同，下面列举每一种算法的原理以及code示例，详细的使用方法参考如下[链接](http://igit.58corp.com/AIgroups/ModelHelper/blob/master/tutorials/feature%20select/feature_select.ipynb)。

创建虚拟数据

```python
import pandas as pd
from sklearn.datasets import make_classification

def make_data():
    data, targets = make_classification(
        n_samples=10000,
        n_features=100,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    cols = [f"feat{i}" for i in range(1, 1+data.shape[1])]
    data = pd.DataFrame(data, columns=cols)
    return data, targets
data, targets = make_data()
```

### 2.1 random search（随机搜索）

每次随机选取部分特征子集（特征选取的数量由参数`sample`控制，默认为None，相当于随机抽取`[min_feature, len(feature)`个特征），通过特征子集效果（交叉验证或者验证集，参数`k_fold`, `create_valid`, `valid_ratio`, `valid_x`, `valid_y`）控制，共迭代`max_iter`次
```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.feature_selection.wrapper import random_search
      
clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择80%的特征，通过3折交叉验证来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = random_search(data, targets, clf, initialize_by_model=True, k_fold=3, sample=0.8, max_iter=10, random_state=666)
```

### 2.2 lvw（Las Vegas Wrapper）

参数设置和随机搜索一致，每次在最优的特征子集下，进行特征子集的随机选择。

> 具体原理可以参考周志华《机器学习》p250-p252

```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.feature_selection.wrapper import lvw

clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择50个的特征，通过产生20%的验证集的auc值来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = lvw(data, targets, clf, initialize_by_model=True, k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score, sample=50, max_iter=10, random_state=667)
```

### 2.3 weight_search(加权选择)

参数设置和随机搜索一致，按照特征重要性进行特征子集的选择。

```python
from sklearn.tree import DecisionTreeClassifier
from skleran.metrics import roc_auc_score
from model_helper.feature_selection.wrapper import weight_search
  
clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择[1, feature_num)个的特征，通过3折交叉验证的平均auc值来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = weight_search(data, targets, clf, initialize_by_model=True, k_fold=3, valid_ratio=0.2, metric_func=roc_auc_score, sample=None, max_iter=10, random_state=667)
```

### 2.4 top_feat_search（特征重要性前百分比选择）

参数略有不同和上面，按照特征重要性前top的比例列表进行特征子集的选择。

```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.feature_selection.wrapper import top_feat_search
  
clf = DecisionTreeClassifier()
# 以全量特征作为初始化的效果，共产生三个特征子集，按照特征重要性进行排序，选择比例为0.9,0.8,0.7，通过产生20%的验证集的auc值来评估特征子集效果
feature_subset, subset_effect = top_feat_search(data, targets, clf, initialize_by_model=True, top_ratio_list=[0.9, 0.8, 0.7], k_fold=None,create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,random_state=667)
```

### 2.5 分布式特征选择

目前该方式集成了上面的随机搜索random search by feature importance和top feature importance这三种方式。使用方式和之前

#### 2.5.1 random search

需要指定`method = "random"`，其余使用方式基本和前面是一致的。

```python
from pyspark.sql import SparkSession
from model_helper.feature_selection.wrapper import distributed_feature_select

clf = DecisionTreeClassifier()
spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
subset, effect = distributed_feature_select(spark, data, targets, clf, method="random", k_fold=3,
                                            max_iter=10, random_state=666)
print(effect, subset)
```

#### 2.5.2 weight search

需要指定`method = "weight"`，其余使用方式和前面一致。

```python
from pyspark.sql import SparkSession
from model_helper.feature_selection.wrapper import distributed_feature_select

clf = DecisionTreeClassifier()
spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
subset, effect = distributed_feature_select(spark, data, targets, clf, method="weight", k_fold=3,
                                            max_iter=10, random_state=666)
print(effect, subset)
```

#### 2.5.3 top feat search

需要指定`method = "top_feat"`，其余使用方式和前面一致。

```python
from pyspark.sql import SparkSession
from model_helper.feature_selection.wrapper import distributed_feature_select

clf = DecisionTreeClassifier()
spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
subset, effect = distributed_feature_select(spark, data, targets, clf, method="top_feat", k_fold=3,
                                            max_iter=10, random_state=666)
print(effect, subset)
```

## 3. 超参数搜索

目前实现了四种方式，分别为随机搜索，网格搜索，bayes_opt和hyper_opt，详细的使用方法参考如下[链接](http://igit.58corp.com/AIgroups/ModelHelper/blob/master/tutorials/param%20search/param_serach.ipynb)。

### 3.1 随机搜索

随机进行参数的搜索

- 单机模式

  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import roc_auc_score
  
  from model_helper.hyper_parameter_tuning import param_search
  
  clf = DecisionTreeClassifier()
  param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
  
  # 随机从参数网格进行参数选择，共迭代10次，每次评估的准则通过产生的20%验证集
  best_effect, best_param = param_search(df, label, clf, param_grid, method="random", k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
  ```

- 分布式模式

  通过spark进行分布式的遍历，相比上面的方法，增加了两个变量，`spark`(Spark Session)和`num_partition`(int or None,默认为None，自动根据资源分配)

  ```python
  from pyspark.sql import SparkSession
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import roc_auc_score
  
  from model_helper.hyper_parameter_tuning import distributed_param_search
  
  clf = DecisionTreeClassifier()
  param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
  
  # 随机从参数网格进行参数选择，共迭代10次，每次评估的准则通过产生的20%验证集
  spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
  
  best_effect, best_param = distributed_param_search(spark, df, label, clf, param_grid, method="random", num_partition=None, k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
  ```

### 3.2 网格搜索

  对所有的网格的可能参数组合遍历

  - 单机模式

    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    
    from model_helper.hyper_parameter_tuning import param_search
    
    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    
    # 从参数网格进行参数选择，遍历所有可能，每次评估的准则通过产生的20%验证集
    best_effect, best_param = param_search(df, label, clf, param_grid, method="grid", k_fold=3, random_state=666, create_valid=True, valid_ratio=0.2)
    ```
    
  - 分布式模式

    通过spark进行分布式的遍历，相比上面的方法，增加了两个变量，`spark`(Spark Session)和`num_partition`(int or None,默认为None，自动根据资源分配)
    
    ```python
    from pyspark.sql import SparkSession
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    
    from model_helper.hyper_parameter_tuning import distributed_param_search
    
    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    
    # 从参数网格进行参数选择，遍历所有可能，每次评估的准则通过产生的20%验证集
    spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
    
    best_effect, best_param = distributed_param_search(spark, df, label, clf, param_grid, method="grid", num_partition=None, k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
    ```

### 3.3 bayes_opt

基于开源的包`bayes_opt`进行了包裹，若使用，保证bayes_opt >= 1.0.1

```python
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier

from model_helper.hyper_parameter_tuning import bayes_search

clf = RandomForestClassifier()
param_space = {"max_features": {"interval": (0.1, 0.9), "type": float},
"n_estimators": {"interval": (10, 250), "type": int},
"min_samples_split": {"interval": (2, 25), "type": int}
}
best_result, best_params = bayes_search(data, targets, model=clf, param_space=param_space, n_iter=10,
k_fold=3, random_state=666)
print(f"best_result is {best_result}, best_param is {best_params}")
```

### 3.4 hyper_opt

基于开源的包`hyper_opt`进行了包裹，若使用，保证hyper_opt >= 0.1.2

```python
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from model_helper.hyper_parameter_tuning import hyperopt_search

clf = RandomForestClassifier()
param_space = {"max_features": hp.uniform("max_features", 0.1, 0.9),
"n_estimators": hp.choice("n_estimators", range(10, 100)),
"min_samples_split": hp.choice("min_samples_split", [2, 10, 100])
}
trials, best_params = hyperopt_search(data, targets, model=clf, param_space=param_space, n_iter=10,
k_fold=3, random_state=666)
for i in trials:
print(i["result"]["stuff"])
print(f"best_param is {best_params}")
```

## 4. 模型解释

本部分为模型解释部分，主要分为两块，机器学习和深度学习部分。下面介绍大致原理和部分示例代码，详细的使用方式，参考[链接](http://igit.58corp.com/AIgroups/ModelHelper/blob/master/tutorials/model%20explain/model_explain.ipynb)。

- 机器学习

      目前提供的解释的模型有XGBoost、LightGBM和RandomForest模型，均是基于`shap`包得到。

- 深度学习

    这块主要分为图像和文本。

      - 图像

        只要模型的输入和输出的形状（中间网络部分可以任意设置）符合标准，那么模型均可以解释（至少目前发现是可以的），对模型的解释体现到图像上每一个像素的得分上，最终将得分以图像的形式进行展示，可以直观看到模型做出预测的依据。

      - 文本

        这部分和图像类似，只要模型的输入和输出的符合标准，那么模型均可以解释，不过对模型的解释体现到文本的每一个单词的得分上，最终可以按照指定的规则将解释输出。

        > 需要注意的是目前Keras版本的BERT模型在测试时失败，原因有可能目前使用的BERT模型的keras版均是非官方私人实现的，后面有待进一步验证

### 4.1 机器学习模型

目前提供的解释的模型有XGBoost、LightGBM和RandomForest模型，对模型的解释主要为对样本预测的概率进行解释，将其分解到每个特征的得分上，特征得分越高则对增益越大，越小则增益越小。

基于训练的模型，对预测样本的得分给出特征维度的解释。

目前实现方式是基于特征的`shap`值，主要采用了四种策略（`max_shap`, `min_shap`,`abs_shap`,`threshold`），还添加了特征分组的概念，下面一一作出解释以及code示例。目前已经实现了`lightgbm`、`xgboost`和`RandomForest`模型的解释。

下面以`lightgbm`模型为例进行讲解（`xgboost`和`RandomForest`模型和此方式一致），

数据构建以及模型训练，

```python
from sklearn.model_selection import train_test_split
import shap

X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)  # 模型训练

pred = model.predict(X_test)  # 模型预测
```

接下来相当于对预测`pred`基于样本`X_test`的特征取值作出解释。

#### 4.1.1 特征选取策略

选取策略部分共有四块，分别为最大shap值，最小shap值，shap绝对值和概率阈值。

##### 4.1.1.1 最大shap值

按照最大shap值策略进行top特征的选取，这种方式选取的特征代表**最大正向促进了最终样本的得分**。

```python
from model_helper.model_explain import lgb_explain

reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap")  # 每个样本取top3的最大shap值所应的特征，并返回特征的取值，将其拼接为字符串
print(reasons[:5])
```

输出如下：

```
['Race value is 4.0,Occupation value is 4.0,Age value is 39.0',
 'Education-Num value is 13.0,Hours per week value is 58.0,Age value is 48.0',
 'Sex value is 1.0,Marital Status value is 2.0,Relationship value is 4.0',
 'Workclass value is 7.0,Occupation value is 4.0,Sex value is 1.0',
 'Education-Num value is 13.0,Relationship value is 4.0,Age value is 50.0']
```

##### 4.1.1.2 最小shap值

按照最小shap值策略进行top特征的选取，这种方式选取的特征代表**最大负向促进了最终样本的得分**。

```python
from model_helper.model_explain import lgb_explain

reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="min_shap")  # 每个样本取top3的最小shap值所应的特征，并返回特征的取值，将其拼接为字符串
print(reasons[:5])
```

输出如下：

```
['Capital Gain value is 0.0,Sex value is 0.0,Relationship value is 1.0',
 'Capital Loss value is 0.0,Capital Gain value is 0.0,Relationship value is 1.0',
 'Hours per week value is 35.0,Capital Gain value is 0.0,Age value is 22.0',
 'Marital Status value is 4.0,Relationship value is 0.0,Age value is 23.0',
 'Workclass value is 6.0,Capital Gain value is 0.0,Hours per week value is 8.0']
```

##### 4.1.1.3 shap绝对值

按照绝对值shap值策略进行top特征的选取，这种方式选取的特征代表**最大贡献促进了最终样本的得分**。

```python
from model_helper.model_explain import lgb_explain

reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="abs_shap")  # 每个样本取top3的绝对值shap值所应的特征，并返回特征的取值，将其拼接为字符串
print(reasons[:5])
```

输出如下：

```
['Occupation value is 4.0,Age value is 39.0,Relationship value is 1.0',
 'Relationship value is 1.0,Hours per week value is 58.0,Age value is 48.0',
 'Marital Status value is 2.0,Relationship value is 4.0,Age value is 22.0',
 'Marital Status value is 4.0,Relationship value is 0.0,Age value is 23.0',
 'Relationship value is 4.0,Hours per week value is 8.0,Age value is 50.0']
```

##### 4.1.1.4 概率阈值

按照预测概率阈值进行top特征选取，如果概率大于设定的阈值，则该样本按照`max_shap`方式进行特征选取，否则按照`min_shap`方式进行特征的选取。

```python
from model_helper.ModelExplain.lgb_explain import lgb_explain

reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="threshold", prob_threshold=0.5)
print(reasons[:5])
```

输出如下 ：

```
['Capital Gain value is 0.0,Sex value is 0.0,Relationship value is 1.0',
 'Education-Num value is 13.0,Hours per week value is 58.0,Age value is 48.0',
 'Hours per week value is 35.0,Capital Gain value is 0.0,Age value is 22.0',
 'Marital Status value is 4.0,Relationship value is 0.0,Age value is 23.0',
 'Workclass value is 6.0,Capital Gain value is 0.0,Hours per week value is 8.0']
```

#### 4.1.2 **特征分组**

特征分组相当于输入特征按照业务或者其它规则分为不同的组，然后从不同的组中选取top shap值的特征，

分组形如`[("f1", "f2", "f3"), ("f4", "f5", "f6"), ("f7", "f8")]`，上面的方式相当于只有一个特征分区，里面包含全量的特征。

```python
from model_helper.model_explain import lgb_explain

feature_group = [("Age", "Sex", "Workclass"), ("Education-Num", "Marital Status", "Occupation", "Capital Loss"), ("Hours per week", "Capital Gain", "Relationship")]

reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap", feature_group=feature_group)
print(reasons[:5])
```

输出如下：

```
['Age value is 39.0,Occupation value is 4.0,Hours per week value is 40.0',
 'Age value is 48.0,Education-Num value is 13.0,Hours per week value is 58.0',
 'Sex value is 1.0,Marital Status value is 2.0,Relationship value is 4.0',
 'Sex value is 1.0,Occupation value is 4.0,Hours per week value is 40.0',
 'Age value is 50.0,Education-Num value is 13.0,Relationship value is 4.0']
```

上面选择的top3的特征，此时由于特征分组数量为3，则会每个组内选择top 1的特征，如果需要选择特征数量大于3个，则会每个组进行均匀分配，另外参数`if_sort_group_by_feature_importance`可以指定给定特征分组的排序，如果设置为True，则会按照特征重要性先把特征分组进行排序，这样显示特征顺序就会按此排列

> 给定特征分组中，可以不用包含全量特征，这样选择的特征就不会有此特征，但是不可以存在重复特征或者不在训练数据中的特征，否则程序会raise ValueError

#### 4.1.3 自定义输出的格式

在上面输出中，默认输出的格式为`特征名+ value is 特征值`的 输出模式，可以自定义拼接方式。

```python
feature_meaning_dic = {i: f"{i}_meaning" for i in X_test.columns}  # 特征名具体含义

def self_func(feature_value_pair, feature_meaning_dic):
    ret = []
    for feature, feature_value in feature_value_pair:
        feature_meaning = feature_meaning_dic.get(feature, feature)  # if no this key meaning, use key instead
        ret.append(f"{feature_meaning}: {feature_value}")
    return ",".join(ret)

feature_group = [("Age", "Sex", "Workclass"), ("Education-Num", "Marital Status", "Occupation", "Capital Loss"), ("Hours per week", "Capital Gain", "Relationship")] # 特征分组
reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap", feature_group=feature_group, feature_meaning_dic=feature_meaning_dic, verbal_express=self_func)

print(reasons[:5])
```

输出如下：

```
['Age_meaning: 39.0,Occupation_meaning: 4.0,Hours per week_meaning: 40.0',
 'Age_meaning: 48.0,Education-Num_meaning: 13.0,Hours per week_meaning: 58.0',
 'Sex_meaning: 1.0,Marital Status_meaning: 2.0,Relationship_meaning: 4.0',
 'Sex_meaning: 1.0,Occupation_meaning: 4.0,Hours per week_meaning: 40.0',
 'Age_meaning: 50.0,Education-Num_meaning: 13.0,Relationship_meaning: 4.0']
```

### 4.2 深度模型解释

以上部分的模型解释主要针对树模型算法，该部分主要介绍深度模型的解释。

#### 4.2.1 图像

以mnist数据集为例（图像分类）为例，

##### 4.2.1.1 构建模型及训练

```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
activation='relu',
input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()
```

output：

```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/1
60000/60000 [==============================] - 81s 1ms/step - loss: 0.2646 - acc: 0.9176 - val_loss: 0.0734 - val_acc: 0.9759
Test loss: 0.07343587589133531
Test accuracy: 0.9759

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               1179776   
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_____________________________
```

##### 4.2.1.2 解释预测样本

```python
from model_helper.model_explain import explain_image_plot

# create backgroud sample, the more the sample number, the more accurate the plot 
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
fig, ax = explain_image_plot(x_test[1:5], model, background)  # 对其中四个样本作出解释
```

output image：

![](./pics/image_explain_plot.png)

其中左侧是挑选四幅测试集图像，右侧是模型作出的预测，以及对应的解释（红色部分表示正向增益，蓝色表示反向增益）

#### 4.2.2 文本

##### 4.2.2.1 构建模型及训练

```python
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```

output:

```
Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 80)
x_test shape: (25000, 80)
Build model...
Train...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1
25000/25000 [==============================] - 17s 663us/step

Test score: 0.38192642364501955
Test accuracy: 0.8348000049591064
```

##### 4.2.2.2 解释预测样本

```python
from model_helper.model_explain import explain_text_model

xx_test = x_test[:2]
words = imdb.get_word_index()
num2word = {}
for w in words.keys():
    num2word[words[w]] = w
num2word[0] = "padding"

r = explain_text_model(texts=xx_test, model=model, 
                       background_sample=x_train[:20], 
                       strategy="abs",
                       top_n_words=2, vocab_dic=num2word)

print(f"共解释了{len(xx_test)}个样本")

for i in range(len(xx_test)):
    print(f"第{i+1}个样本：")
    print(r[i])
```

output:

```
共解释了2个样本
第1个样本：
(['world', 'sequence'], [0.055808417630123584, 0.04994702937033253], 0)
第2个样本：
(['realistic', '\x96'], [0.09124832981685813, -0.06742123246962137], 0)
```

相当于对测试集前两个样本的预测作出解释，返回是一个长度为2的list，每个元素是一个三元组，第一个为输入的文本，第二个元组为对应文本的shap值，第三个元组为模型预测类别位置。解释的规则是按照最大shap绝对值的前2个单词（可以按照最小shap值和最大shap值，亦或是将全部输入样本进行解释）。具体的使用参数可以参考函数的参数注释部分。

## 5. 模型融合

本部分主要分为三块，分别是Stacking，Blending和Bagging。这几部分所有实现均包含了单机版本和分布式版本，下面的介绍中主要从这两方面介绍，详细的使用方式，参考[链接](http://igit.58corp.com/AIgroups/ModelHelper/blob/master/tutorials/model%20ensemble/model_ensemble.ipynb)。

### 5.1 Stacking

Stacking模型是指将多种分类器组合在一起来取得更好表现的一种集成学习模型。一般情况下，Stacking模型分为两层。第一层中我们训练多个不同的模型，然后再以第一层训练的各个模型的输出作为输入来训练第二层的模型，以得到一个最终的输出。可参考[文章](https://blog.csdn.net/data_scientist/article/details/78900265)

#### 5.1.1 单机版本

以二分类为例（多分类目前不支持），

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_helper.model_ensemble import StackingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

clf = StackingClassifier(k_fold=5, base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),DecisionTreeClassifier()], meta_learner=LogisticRegression())  # 构建学习器实例

clf.fit(X=train_x, y=train_y)  # 训练

pred = clf.predict_proba(X=test_x)[:, 1]  # 预测
auc_val = roc_auc_score(y_true=test_y, y_score=pred)  # 评估
print(auc_val)
```

以回归任务为例

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_helper.model_ensemble import StackingRegressor

X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

clf = StackingRegressor(k_fold=5, base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
DecisionTreeRegressor()],
meta_learner=LinearRegression())
clf.fit(X=train_x, y=train_y)

pred = clf.predict(X=test_x)
r2_val = r2_score(y_true=test_y, y_pred=pred)
print(r2_val)
```

#### 5.1.2 分布式版本

分布式版本和上面几乎一致，需要额外指定两个参数`distribute`和`spark`

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession

from model_helper.model_ensemble import StackingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

spark = SparkSession.builder.appName("distribute").enableHiveSupport().getOrCreate()

clf = StackingClassifier(k_fold=5, base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),DecisionTreeClassifier()], meta_learner=LogisticRegression(), distribute=True, spark=spark)  # 构建学习器实例

clf.fit(X=train_x, y=train_y)  # 训练

pred = clf.predict_proba(X=test_x)[:, 1]  # 预测
auc_val = roc_auc_score(y_true=test_y, y_score=pred)  # 评估
print(auc_val)
```

回归任务不再展开，同样指定上面提到的两个参数即可

### 5.2 Blending

Blending与Stacking大致相同，只是Blending的主要区别在于训练集不是通过k-fold的CV策略来获得预测值从而生成第二阶段模型的特征，而是建立一个Holdout集，例如10%的训练数据。

#### 5.2.1 单机版本

以二分类为例（多分类目前不支持），

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_helper.model_ensemble import BlendingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

clf = BlendingClassifier(base_train_size=0.8, base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),DecisionTreeClassifier()], meta_learner=LogisticRegression())  # 构建学习器实例

clf.fit(X=train_x, y=train_y)  # 训练

pred = clf.predict_proba(X=test_x)[:, 1]  # 预测
auc_val = roc_auc_score(y_true=test_y, y_score=pred)  # 评估
print(auc_val)
```

以回归任务为例，

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_helper.model_ensemble import BlendingRegressor

X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

clf = BlendingRegressor(base_train_size=0.8, base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
DecisionTreeRegressor()],
meta_learner=LinearRegression())
clf.fit(X=train_x, y=train_y)

pred = clf.predict(X=test_x)
r2_val = r2_score(y_true=test_y, y_pred=pred)
print(r2_val)
```

#### 5.2.2 分布式版本

分布式版本和上面几乎一致，需要额外指定两个参数`distribute`和`spark`

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession

from model_helper.model_ensemble import BlendingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

spark = SparkSession.builder.appName("distribute").enableHiveSupport().getOrCreate()

clf = BlendingClassifier(base_train_size=0.8, base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),DecisionTreeClassifier()], meta_learner=LogisticRegression(), distribute=True, spark=spark)  # 构建学习器实例

clf.fit(X=train_x, y=train_y)  # 训练

pred = clf.predict_proba(X=test_x)[:, 1]  # 预测
auc_val = roc_auc_score(y_true=test_y, y_score=pred)  # 评估
print(auc_val)
```

### 5.3 Bagging

Bagging同样是属于模型集成的一种方式，不同于Stacking和Blending的两阶段训练，Bagging只需要一阶段的训练，然后将一阶段的模型预测结果集成即可。

#### 5.3.1 单机版本

以二分类为例（多分类目前不支持）

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_helper.model_ensemble import BaggingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(), DecisionTreeClassifier()], predict_strategy="mean")
clf.fit(X=train_x, y=train_y)

pred = clf.predict_proba(X=test_x)[:, 1]
auc_val = roc_auc_score(y_true=test_y, y_score=pred)
print(auc_val)
```

以回归任务为例

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_helper.model_ensemble import BaggingRegressor

X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# test1 test normal process
clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(), DecisionTreeRegressor()], predict_strategy="mean")
clf.fit(X=train_x, y=train_y)

pred = clf.predict(X=test_x)
r2_val = r2_score(y_true=test_y, y_pred=pred)
print(r2_val)
```

#### 5.3.2 分布式版本

分布式版本和上面几乎一致，需要额外指定两个参数`distribute`和`spark`

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession

from model_helper.model_ensemble import BaggingClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

spark = SparkSession.builder.appName("distribute").enableHiveSupport().getOrCreate()

clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),DecisionTreeClassifier()], meta_learner=LogisticRegression(), predict_strategy="mean", distribute=True, spark=spark)  # 构建学习器实例

clf.fit(X=train_x, y=train_y)  # 训练

pred = clf.predict_proba(X=test_x)[:, 1]  # 预测
auc_val = roc_auc_score(y_true=test_y, y_score=pred)  # 评估
print(auc_val)
```

## TO DO LIST

- add class `FeatureEngineering`


