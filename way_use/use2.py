import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

# 加载LDA模型和CountVectorizer
with open('../model_use/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)
with open('../model_use/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 假设texts目录下有多个txt文件，我们分析第一个文件
texts_dir = '../texts'  # 你的texts目录
file_paths = [os.path.join(texts_dir, f) for f in os.listdir(texts_dir) if f.endswith('.txt')]
file_path = file_paths[0]  # 或指定你要分析的txt文件的路径

# 读取txt文件内容
with open(file_path, 'r', encoding='utf-8') as f:  # 根据你的文件编码调整encoding参数
    text = f.read()

# 预处理文本（如果需要的话，例如分词、去除停用词等）
# 注意：这里的预处理应该与训练模型时保持一致
# 如果是中文文本，并且使用了jieba分词，则需要先分词
# import jieba
# segments = jieba.cut(text)
# preprocessed_text = ' '.join(segments)

# 假设文本已经是预处理过的（分词、去除停用词等）
preprocessed_text = text  # 这里只是一个示例，实际使用时应该进行预处理

# 使用CountVectorizer将文本转换为特征向量
tf_vector = vectorizer.transform([preprocessed_text])

# 使用LDA模型预测文本的主题分布
topic_distribution = lda_model.transform(tf_vector)

# 解读主题分布，确定文本的主要主题
# 通常，主要主题是具有最高概率的主题
main_topic = topic_distribution[0].argmax()
print(f"文本的主要主题是: {main_topic}")


# 如果你想查看该主题下的关键词
def print_topic_keywords(model, feature_names, topic_idx, top_n=10):
    print(f"主题 {topic_idx} 的关键词:")
    print(" ".join([feature_names[i] for i in model.components_[topic_idx].argsort()[:-top_n - 1:-1]]))


print_topic_keywords(lda_model, vectorizer.get_feature_names_out(), main_topic)