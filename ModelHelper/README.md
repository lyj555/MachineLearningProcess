# Model Helper

The purpose of the project is to make modelers more efficient.

目录
--

- 1._Installation
- 2._特征选择

    - 2.1 random_search（随机搜索）
    
    - 2.2 lvw（Las Vegas Wrapper）
    
    - 2.3_random search by feature importance(通过特征重要性做特征筛选)
    - 2.4_top feature importance
- 3._超参数选择
    - 3.1 随机选择
    
    - 3.2 网格搜索
    
    - 3.3 bayes_opt
    
    - 3.4 hyper_opt
- 4._模型解释
  
    - 4.1 特征选取策略
        - 4.1.1 最大shap值
        - 4.1.2 最小shap值
        - 4.1.3 shap绝对值
        - 4.1.4 概率阈值
    - 4.2 特征分组
    - 4.3 自定义输出的格式
- TO DO LIST




## 1. Installation

- 直接使用

    直接复制文件夹`model_helper`至本地项目， 然后import使用即可

- 本地安装

    - `git clone https://github.com/lyj555/MachineLearningProcess.git` 

    - `python setup.py install`



## 2. 特征选择

目前特征选择部分实现了四种算法，分别为**随机搜索**、**lvw**、**random search by feature importance**和**top feature importance**，这几种算法本质区别主要是每次特征子集的选取不同，下面列举每一种方式code示例。

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

每次随机选取部分特征子集（由参数`sample`控制），通过特征子集效果（交叉验证或者验证集，参数`k_fold`, `create_valid`, `valid_ratio`, `valid_x`, `valid_y`）控制，共迭代`max_iter`次
```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.FeatureSelection.wrapper import random_search
      
clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择80%的特征，通过3折交叉验证来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = random_search(data, targets, clf, initialize_by_model=True, k_fold=3, sample=0.8, max_iter=10, random_state=666)
```

### 2.2 lvw（Las Vegas Wrapper）

参数设置和随机搜索一致，每次在最优的特征子集下，进行特征子集的随机选择。
```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.FeatureSelection.wrapper import lvw

clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择50个的特征，通过产生20%的验证集的auc值来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = lvw(data, targets, clf, initialize_by_model=True, k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score, sample=50, max_iter=10, random_state=667)
```

### 2.3 random search by feature importance(通过特征重要性做特征筛选)

参数设置和随机搜索一致，按照特征重要性进行特征子集的选择。

```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.FeatureSelection.wrapper import random_search_by_model_feat
  
clf = DecisionTreeClassifier()
# 以全量特征作为初始化效果，随机选择[1, feature_num)个的特征，通过3折交叉验证的平均auc值来评估特征子集效果，迭代10次进行比较筛选
feature_subset, subset_effect = random_search_by_model_feat(data, targets, clf, initialize_by_model=True, k_fold=3, valid_ratio=0.2, metric_func=roc_auc_score, sample=None, max_iter=10, random_state=667)
```

### 2.4 top feature importance

参数略有不同和上面，按照特征重要性前top的比例列表进行特征子集的选择

```python
from sklearn.tree import DecisionTreeClassifier
from model_helper.FeatureSelection.wrapper import top_feat_by_model
  
clf = DecisionTreeClassifier()
# 以全量特征作为初始化的效果，共产生三个特征子集，按照特征重要性进行排序，选择比例为0.9,0.8,0.7，通过产生20%的验证集的auc值来评估特征子集效果
feature_subset, subset_effect = top_feat_by_model(data, targets, clf, initialize_by_model=True, top_ratio_list=[0.9, 0.8, 0.7], k_fold=None,create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,random_state=667)
```

> 每种算法的参数基本是相同的。

## 3. 超参数搜索

目前实现了四种方式，分别为随机搜索，网格搜索，bayes_opt和hyper_opt

### 3.1 随机搜索

随机进行参数的搜索

- 单机模式

  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import roc_auc_score
  
  from model_helper.HyperparameterTuning.param_search import param_search
  
  clf = DecisionTreeClassifier()
  param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
  
  # 随机从参数网格进行参数选择，共迭代10次，每次评估的准则通过产生的20%验证集
  best_effect, best_param = param_search(df, label, clf, param_grid, method="random", k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
  ```

- 分布式模式

  通过spark进行分布式的遍历，相比上面的方法，增加了两个变量，`spark`(Spark Session)和`num_partition`(int or None,默认为None，自动根据资源分配)

  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import roc_auc_score
  
  from model_helper.HyperparameterTuning.param_search import distributed_param_search
  
  clf = DecisionTreeClassifier()
  param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
  
  # 随机从参数网格进行参数选择，共迭代10次，每次评估的准则通过产生的20%验证集
  best_effect, best_param = distributed_param_search(spark, df, label, clf, param_grid, method="random", num_partition=None, k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
  ```

### 3.2 网格搜索

  对所有的网格的可能参数组合遍历

  - 单机模式

    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    
    from model_helper.HyperparameterTuning.param_search import param_search
    
    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    
    # 从参数网格进行参数选择，遍历所有可能，每次评估的准则通过产生的20%验证集
    best_effect, best_param = param_search(df, label, clf, param_grid, method="grid", k_fold=3, random_state=666, create_valid=True, valid_ratio=0.2)
    
    ```

  - 分布式模式

    通过spark进行分布式的遍历，相比上面的方法，增加了两个变量，`spark`(Spark Session)和`num_partition`(int or None,默认为None，自动根据资源分配)
    
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    
    from model_helper.HyperparameterTuning.param_search import distributed_param_search
    
    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    
    # 从参数网格进行参数选择，遍历所有可能，每次评估的准则通过产生的20%验证集
    best_effect, best_param = distributed_param_search(spark, df, label, clf, param_grid, method="grid", num_partition=None, k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
    ```

### 3.3 bayes_opt

基于开源的包`bayes_opt`进行了包裹，若使用，保证bayes_opt >= 1.0.1

```python
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier

from model_helper.HyperparameterTuning.bayes_opt import bayes_search

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
from model_helper.HyperparameterTuning.hyper_opt import hyperopt_search

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

基于训练的模型，对预测样本的得分给出特征维度的解释。

目前实现方式是基于特征的`shap`值，主要采用了四种策略（`max_shap`, `min_shap`,`abs_shap`,`threshold`），还添加了特征分组的概念，下面一一作出解释以及code示例。目前已经实现了`lightgbm`和`xgboost`模型的解释。

下面以`lightgbm`模型为例进行讲解（`xgboost`模型和此方式一致），

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

### 4.1 特征选取策略

#### 4.1.1 最大shap值

按照最大shap值策略进行top特征的选取，这种方式选取的特征代表**最大正向促进了最终样本的得分**。

```python
from model_helper.ModelExplain.lgb_explain import lgb_explain

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

#### 4.1.2 最小shap值

按照最大shap值策略进行top特征的选取，这种方式选取的特征代表**最大负向促进了最终样本的得分**。

```python
from model_helper.ModelExplain.lgb_explain import lgb_explain

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

#### 4.1.3 shap绝对值

按照绝对值shap值策略进行top特征的选取，这种方式选取的特征代表**最大贡献促进了最终样本的得分**。

```python
from model_helper.ModelExplain.lgb_explain import lgb_explain

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

#### 4.1.4 概率阈值

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

### 4.2 **特征分组**

特征分组相当于输入特征按照业务或者其它规则分为不同的组，然后从不同的组中选取top shap值的特征，

分组形如`[("f1", "f2", "f3"), ("f4", "f5", "f6"), ("f7", "f8")]`，上面的方式相当于只有一个特征分区，里面包含全量的特征。

```python
from model_helper.ModelExplain.lgb_explain import lgb_explain

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

### 4.3 自定义输出的格式

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



## TO DO LIST

- add class `FeatureEngineering`