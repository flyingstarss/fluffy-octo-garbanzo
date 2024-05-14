import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# 读取数据集
df = pd.read_csv('/data/segmented_texts.csv', encoding='utf-8')  # 根据你的数据集情况可能需要调整encoding参数
texts = df['segmented_text'].tolist()  # 假设'text_column_name'是包含文本的列名

# 初始化CountVectorizer（这里可以根据需要调整参数）
vectorizer = CountVectorizer(stop_words='english')  # 如果你处理的是英文文本，否则使用中文停用词列表

# 将文本数据转换为词频矩阵
X = vectorizer.fit_transform(texts)

# 初始化LDA模型（设置主题数量和其他参数）
lda_model = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', random_state=0)

# 训练LDA模型
lda_model.fit(X)

# 保存LDA模型
with open('../model_false/lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)

# 保存CountVectorizer（如果需要后续用于新文本的转换）
with open('../model_false/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# 打印完成信息
print("LDA模型训练完成并已保存。")
