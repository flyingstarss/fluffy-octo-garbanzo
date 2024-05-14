import os
import pandas as pd

# 初始化一个空的DataFrame来存储所有文件的内容
all_texts = []

# 遍历cleaned_zuowen目录下的所有txt文件
for filename in os.listdir('/data/cleaned_zuowen'):
    if filename.endswith('.txt'):
        file_path = os.path.join('/data/cleaned_zuowen', filename)
        with open(file_path, 'r', encoding='utf-8') as file:  # 假设文件是utf-8编码
            content = file.read()
            # 假设我们只需要将文件名和内容存储在CSV中
            all_texts.append({'file_name': filename, 'content': content})

        # 将列表转换为DataFrame
df = pd.DataFrame(all_texts)

# 将DataFrame保存为CSV文件
df.to_csv('D://App/人工智能导论/word/data/combined_zuowen.csv', index=False, encoding='utf-8-sig')  # 使用utf-8-sig以支持Excel打开

print('文件合并完成，已保存为word/data/combined_zuowen.csv')