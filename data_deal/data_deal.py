import os
import re
import string

# 停用词列表
stop_words = set(
    ['的', '是', '在', '了', '我', '不', '有', '和', '人', '都', '上', '大', '个', '要', '到', '很', '说', '看', '你'])


# 清洗文本的函数
def clean_text(text):
    # 去除HTML标签
    clean_text = re.sub('<[^<]+?>', '', text)
    # 去除标点符号
    translator = str.maketrans('', '', string.punctuation)
    clean_text = clean_text.translate(translator)
    # 转换为小写
    clean_text = clean_text.lower()
    # 去除停用词
    words = clean_text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


# 遍历data/zuowen目录下的所有txt文件
for filename in os.listdir('/data/zuowen'):
    if filename.endswith('.txt'):
        with open(f'D:/App/人工智能导论/word/data/zuowen/{filename}', 'r', encoding='utf-8') as file:
            content = file.read()
            cleaned_content = clean_text(content)

            # 将清洗后的内容保存到新的txt文件，这里假设保持原文件名，但添加_cleaned后缀
        with open(f'D:/App/人工智能导论/word/data/cleaned_zuowen/{filename.replace(".txt", "_cleaned.txt")}', 'w',
                  encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_content)

            # 注意：上面的代码假设data/cleaned_zuowen目录已经存在，如果不存在，你需要先创建它