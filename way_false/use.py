import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 假设你的文本文件位于名为'texts'的目录下
text_directory = 'D://App/人工智能导论/word/texts'
all_filenames = []  # 用于存储文件名的列表
all_texts = []  # 用于存储文本内容的列表

# 遍历文本目录中的所有文件
for filename in os.listdir(text_directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(text_directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            # 读取文件内容并添加到列表中
            text = file.read()
            all_texts.append(text)
            all_filenames.append(filename)  # 添加文件名到列表中

# 加载已保存的CountVectorizer和LDA模型
with open('../model_false/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('../model_false/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

# 将所有文本转换为词频向量
texts_vectorized = vectorizer.transform(all_texts)

# 获取所有文本在各个主题上的分布
topic_distributions = lda_model.transform(texts_vectorized)

# 打印每个文本的主题分布
for i, topic_dist in enumerate(topic_distributions):
    print(f"文本 {i + 1}（文件名: {all_filenames[i]}）的主题分布:")
    for topic_idx, prob in enumerate(topic_dist):
        print(f"主题 {topic_idx}: {prob:.4f}")
    print()

# 确定每个文本的主要主题
main_topics = topic_distributions.argmax(axis=1)
for i, main_topic in enumerate(main_topics):
    print(f"文本 {i + 1}（文件名: {all_filenames[i]}）的主要主题是: {main_topic}")