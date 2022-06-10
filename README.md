# Chinese-Text-Classification

中文文本分类，基于pytorch，开箱即用。

- 神经网络模型：TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer

- 预训练模型：Bert，ERNIE



## 介绍

### 神经网络模型

模型介绍、数据流动过程：[参考](https://zhuanlan.zhihu.com/p/73176084)  

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

| 模型        | 介绍                              |
| ----------- | --------------------------------- |
| TextCNN     | Kim 2014 经典的CNN文本分类        |
| TextRNN     | BiLSTM                            |
| TextRNN_Att | BiLSTM+Attention                  |
| TextRCNN    | BiLSTM+池化                       |
| FastText    | bow+bigram+trigram， 效果出奇的好 |
| DPCNN       | 深层金字塔CNN                     |
| Transformer | 效果较差                          |

### 预训练模型

| 模型       | 介绍                                                         | 备注         |
| ---------- | ------------------------------------------------------------ | ------------ |
| bert       | 原始的bert                                                   |              |
| ERNIE      | ERNIE                                                        |              |
| bert_CNN   | bert作为Embedding层，接入三种卷积核的CNN                     | bert + CNN   |
| bert_RNN   | bert作为Embedding层，接入LSTM                                | bert + RNN   |
| bert_RCNN  | bert作为Embedding层，通过LSTM与bert输出拼接，经过一层最大池化层 | bert + RCNN  |
| bert_DPCNN | bert作为Embedding层，经过一个包含三个不同卷积特征提取器的region embedding层，可以看作输出的是embedding，然后经过两层的等长卷积来为接下来的特征抽取提供更宽的感受眼，（提高embdding的丰富性），然后会重复通过一个1/2池化的残差块，1/2池化不断提高词位的语义，其中固定了feature_maps,残差网络的引入是为了解决在训练的过程中梯度消失和梯度爆炸的问题。 | bert + DPCNN |

参考：

- [ERNIE - 详解](https://baijiahao.baidu.com/s?id=1648169054540877476)
- [DPCNN 模型详解](https://zhuanlan.zhihu.com/p/372904980)
- [从经典文本分类模型TextCNN到深度模型DPCNN](https://zhuanlan.zhihu.com/p/35457093)

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX  
~~pytorch_pretrained_bert~~(预训练代码也上传了, 不需要这个库了)  


## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。数据以字为单位输入模型。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换数据集
 - 按照THUCNews数据集的格式来格式化自己的中文数据集。  
 - 对于神经网络模型：
   - 如果用字，按照数据集的格式来格式化你的数据。  
    - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
    - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。


## 实验效果

机器：一块2080Ti ， 训练时间：30分钟。  

模型|acc|备注
--|--|--
TextCNN|91.22%|Kim 2014 经典的CNN文本分类
TextRNN|91.12%|BiLSTM
TextRNN_Att|90.90%|BiLSTM+Attention
TextRCNN|91.54%|BiLSTM+池化
FastText|92.23%|bow+bigram+trigram， 效果出奇的好
DPCNN|91.25%|深层金字塔CNN
Transformer|89.91%|效果较差
bert|94.83%|单纯的bert
ERNIE|94.61%|说好的中文碾压bert呢  
bert_CNN|94.44%|bert + CNN  
bert_RNN|94.57%|bert + RNN  
bert_RCNN|94.51%|bert + RCNN  
bert_DPCNN|94.47%|bert + DPCNN  

原始的bert效果就很好了，把bert当作embedding层送入其它模型，效果反而降了，之后会尝试长文本的效果对比。

## 预训练语言模型
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

预训练模型下载地址：  
bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz  
              词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
来自[这里](https://github.com/huggingface/pytorch-transformers)   
备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese: https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw  

来自[这里](https://github.com/nghuyong/ERNIE-Pytorch)  

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。  

## 使用说明

### 神经网络方法

```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Att
python run.py --model TextRNN_Att

# TextRCNN
python run.py --model TextRCNN

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

# Transformer
python run.py --model Transformer
```

### 预训练方法

下载好预训练模型就可以跑了：
```
# 预训练模型训练并测试：
# bert
python pretrain_run.py --model bert

# bert + 其它
python pretrain_run.py --model bert_CNN

# ERNIE
python pretrain_run.py --model ERNIE
```

### 预测

预训练模型：

```
# bert (+其他)
python bert_predict.py

# ERNIE
python ERNIE_predict.py
```

神经网络模型：

```
Todo：
```


### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。

## 参考

### 论文

[1] Convolutional Neural Networks for Sentence Classification  

[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  

[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  

[4] Recurrent Convolutional Neural Networks for Text Classification  

[5] Bag of Tricks for Efficient Text Classification  

[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  

[7] Attention Is All You Need

[8] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

[9] ERNIE: Enhanced Representation through Knowledge Integration  

### 仓库

本项目基于以下仓库继续开发优化：

- https://github.com/649453932/Chinese-Text-Classification-Pytorch
- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

