import re

def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 保留中英文数字和空格
    return text.strip()

with open('sample.txt', 'r', encoding='utf-8') as f:
    lines = [clean_text(line) for line in f if line.strip()]

with open('sample.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))