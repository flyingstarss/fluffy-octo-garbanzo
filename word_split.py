import os
import jieba
import pandas as pd
from collections import Counter


# 加载停用词列表
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords


# 分词并去除停用词
def segment_text(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return ' '.join(filtered_words)  # 返回分词后的字符串，用空格分隔


# 处理文件并收集数据
def process_files(input_dir, output_csv, stopwords_path):
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                segmented_text = segment_text(content, load_stopwords(stopwords_path))
                # 将文件名和分词后的文本添加到数据中
                data = {'filename': filename, 'segmented_text': segmented_text}
                all_data.append(data)

                # 将数据保存为CSV文件
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, encoding='utf_8_sig')  # 使用utf_8_sig编码防止中文乱码


# 设置文件路径
input_directory = 'D://App/人工智能导论/word/data/zuowen'
output_csv_file = '/data/segmented_texts.csv'
stopwords_path = '/data/stopwords.txt'

# 调用函数处理文件
process_files(input_directory, output_csv_file, stopwords_path)