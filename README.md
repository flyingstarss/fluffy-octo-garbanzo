# fluffy-octo-garbanzo
蓬松的八角鹰嘴豆——文章主题分析

# 项目简介
本项目是基于LDA模型的文章主题分析。

# LDA模型简介
LDA模型，也称为Latent Dirichlet Allocation（潜在狄利克雷分配）模型，是一种文档主题生成模型，也称为三层贝叶斯概率模型，包含词、主题和文档三层结构。LDA模型主要用于识别大规模文档集或语料库中潜藏的主题信息。

LDA模型的基本思想是，一篇文章的每个词都是通过以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语。因此，LDA可以生成“文档-主题”和“主题-单词”两个概率分布。

LDA模型是一种非监督机器学习技术，它采用了词袋（bag of words）的方法，将每一篇文档视为一个词频向量，从而将文本信息转化为易于建模的数字信息。每一篇文档代表了一些主题所构成的一个概率分布，而每一个主题又代表了很多单词所构成的一个概率分布。

LDA模型的应用非常广泛，包括但不限于信息检索、情感分析、广告推荐、新闻质量分类、短文本-短文本语义匹配、新闻个性化推荐等。通过LDA模型，我们可以自动归纳出一些相关联的单词并组成主题，减少数据量，从而更好地处理大规模数据集。

总的来说，LDA模型是一种强大的工具，可以帮助我们识别和理解文档的主题结构，为各种文本挖掘和机器学习任务提供有力的支持。

# 环境配置及工具使用
python3.11
pandas
sklearn
pickle
os
jieba
以及相关的pytorch配置

# 文件名称含义
data:用于存放数据集和分词表，包括预处理过后的数据集
data_deal:存放数据处理方法，包括处理原始数据，去除标点符号，使用分词表分割文章。
model_use:用于存放生成的模型
texts:用于存放需要分析主题的文章
way_use:用于存放训练方法和使用方法

# 数据集来源

# 注意事项
路径：代码中使用的路径部分为绝对路径，使用时需要改为自己的路径。
结果：由于参数设置原因，仅设置了10个主题集，每个主题集合之中包含多个主题，对文章进行主题分析时，并不会指出其具体的主题，而是给出其主题所在的主题集。
使用：数据集与训练过的模型均未上传，其他人使用则需要再次下载数据集并训练。



