
# framework

#### data/ 文件夹包含训练和测试数据文件以及图像文件。

#### models/ 文件夹用于存储模型定义和训练过程中的检查点。
 ｜---OTE：选用output transformer encoder 方法

#### utils/ 文件夹包含数据集和训练辅助函数的实用程序脚本。
 ｜---API:写入4个api分别进行编码解码，文本处理 BERT的tokenizer,图像处理 torchvision的transforms

   &emsp;&emsp;&emsp;&emsp; ｜--decode \
   &emsp;&emsp;&emsp;&emsp; ｜--encode \
   &emsp;&emsp;&emsp;&emsp; ｜--metric \
   &emsp;&emsp;&emsp;&emsp; ｜--dataset

 ｜---dataset.py：将数据转化成json文件并进行基本变换并输出
#### config.py 用于定义一些常规参数

#### train.py 用于训练模型的脚本。

#### main.py 用于在测试集上进行情感标签预测的脚本。

### requirements
chardet==4.0.0
Pillow==9.2.0
scikit_learn==1.1.1
torch==1.9.0
torchvision==0.10.0
tqdm==4.63.0
transformers==4.18.0

### train
(也可直接运行main.py 默认选项为TextModel: roberta-base, ImageModel: ResNet50, FuseModel: OTE)

```shell 
python main.py --do_train --epoch 10 --text_pretrained_model roberta-base --fuse_model_type OTE $单模态(--text_only --img_only)
```

### test

```shell 
python main.py --do_test --text_pretrained_model roberta-base --fuse_model_type OTE --load_model_path $your_model_path$ 单模态(--text_only --img_only)
```

### result

|Feature| acc    |
|---|--------|
|OTE | 75.184 |
|Text-only| 71.856 |
|Imagine-only | 62.384 |


### reference

1.BERT-based Joint Learning for Multimodal Emotion Recognition and Sentiment Analysis（2020）- 这篇论文提出了一种基于OutputTransformerEncoder的联合学习方法，该方法使用BERT模型进行多模态特征编码，并结合情感标签进行联合训练。

2.Multimodal Sentiment Analysis with Word-Level Fusion and Reinforcement Learning（2021）- 这篇文论强化了OTE方法，该方法通过词级融合和强化学习机制，有效地融合了文本、音频和视频等多模态特征进行情感分类。


