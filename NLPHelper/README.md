# NLPHelper

本项目主要三个目的，

1. 封装NLP中常用的功能，比如创建词典，文本转化为ID，数据做上下采样，构造数据iterator，训练的early_stopping_rounds等
2. 封装一些常用的网络结构，比如TextCNN，CRF等
3. 汇总各类NLP任务的通用开发流程，比如文本本类、命名实体识别等

> 以上功能分为pytorch和tensorflow

## 1. NLP常用功能

### 1.1 预处理类

- `build_vocab_by_raw_file`

  构建词典，key是token(字或者词)，value是id。形如{“我”: 1, "你": 2}等，具体的使用参数参考代码介绍。

- `content_to_id`

  将文本转换为ID，即利用上面构建的词典将文本装换为指定长度的文本ids。

- `split_data_with_index`

  数据切分，输入的数据量返回每部分的index，可以指定label，按照label的分布进行切分。

- `sample_data_by_label`

  按照标签进行采样，可以进行上下采样。

### 1.2 模型训练类

- 组建数据的iterator

  将数据按照batch_size进行获取，目前有自己构建的iterator和引用的torch_iterator。

- `train`

  模型训练，可以指定loss，optimizer，early_stopping_batch以及early_stopping_epoch等。

## 2. 网络结构

### 2.1 TextCNN



### 2.2 Bi-LSTM



### 2.3 CRF



## 3. NLP任务

一些常用的NLP任务

### 3.1 分类

以cluemark中数据为例，输入数据为

```json
{"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物", "keywords": "江疏影,美少女,经纪人,甜甜圈"}
```

模型需要的数据格式为：

"江疏影,美少女,经纪人,甜甜圈；江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"    2

> 两列，第一列为文本，第二列为标签（0-n）

评测标准是acc

### 3.2 命名实体识别

以cluemark中数据为例，输入数据为

```json
{"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，", "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}}
```

格式较为复杂，模型需要的数据为

"彭 小 军 认 为 ， 国 内 银 行 现 在 走 的 是 台 湾 的 发 卡 模 式 ， 先 通 过 跑 马 圈 地 再 在 圈 的 地 里 面 选 择 客 户 ，"

"B-name I-name I-name O O O O O O O O O O O O B-address I-address O O O O O O O O O O O O O O O O O O O O O O O O O"

> 两列，一列是原始文本按照空格分割开，第二列是标注，和原始文本等长度，B-tag表示实体的起始位置，I-tag表示实体的中间部分

评测指标为f1值

真实实体数量/识别实体数量/识别正确实体数量

## 4. TO DO LIST

