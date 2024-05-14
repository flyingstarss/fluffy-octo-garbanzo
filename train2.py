import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import jieba  # 如果你处理的是中文文本，需要jieba分词库

# 读取数据集
df = pd.read_csv('../data/segmented_texts.csv', encoding='utf-8')  # 假设数据集是中文的
texts = df['segmented_text'].tolist()  # 假设'segmented_text'是包含已分词的文本列名

# 如果文本还没有分词，使用jieba进行分词
# texts = [' '.join(jieba.cut(text)) for text in texts]

# 初始化CountVectorizer或TfidfVectorizer（这里使用TfidfVectorizer可能会得到更好的主题区分效果）
# 注意：这里需要提供一个中文停用词列表，或者注释掉stop_words参数
# vectorizer = CountVectorizer(stop_words='your_chinese_stopwords_list.txt')
vectorizer = TfidfVectorizer()

# 将文本数据转换为TF-IDF矩阵
X = vectorizer.fit_transform(texts)

# 初始化LDA模型（设置主题数量和其他参数）
lda_model = LatentDirichletAllocation(n_components=10, max_iter=20, learning_method='online', random_state=0)

# 训练LDA模型
lda_model.fit(X)


# 打印每个主题的前N个关键词
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "主题 {}: ".format(topic_idx)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


print("\n主题关键词:")
print_top_words(lda_model, vectorizer.get_feature_names_out(), 10)

# 保存LDA模型和CountVectorizer
with open('../model_use/lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)
with open('../model_use/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("LDA模型训练完成并已保存。")