import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, RegexpParser
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import os
import json
from pathlib import Path

# 下载NLTK所需资源
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# 确保NLTK资源已下载
download_nltk_resources()

# 句法分析规则
grammar = r"""
    NP: {<DT>?<JJ>*<NN|NNS|NNP|NNPS>+}   # 名词短语: 可选限定词 + 0或多个形容词 + 1个或多个名词
    VP: {<VB.*><NP|PP|CLAUSE>*}          # 动词短语: 以动词开头，后面跟 NP, PP 或 CLAUSE
    PP: {<IN><NP>}                       # 介词短语: 介词 + 名词短语
    CLAUSE: {<NP><VP>}                   # 从句: 名词短语 + 动词短语
    CONJ: {<CC><NP|VP|PP|CLAUSE>+}
"""

chunk_parser = nltk.RegexpParser(grammar)

def tokenize_and_tag(text):
    """将文本分割成句子，并进行词性标注"""
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
    return tagged_sentences

def segment_sentence(tagged_sentence):
    """使用句法分析器分割句子"""
    tree = chunk_parser.parse(tagged_sentence)
    segments = []
    current_segment = []

    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            current_segment.extend(subtree.leaves())
            if subtree.label() in {"NP", "VP", "PP", "CLAUSE"}:
                segments.append(" ".join(word for word, tag in current_segment))
                current_segment = []
        else:
            current_segment.append(subtree)

    if current_segment:
        segments.append(" ".join(word for word, tag in current_segment))

    return segments

def format_segments(segments):
    """格式化分割后的片段"""
    return " | ".join(segments) + " |"

def process_text(text):
    """处理文本，返回格式化后的分割结果"""
    tagged_sentences = tokenize_and_tag(text)
    all_segments = []

    for tagged_sentence in tagged_sentences:
        segments = segment_sentence(tagged_sentence)
        all_segments.extend(segments)

    return format_segments(all_segments)

# 预处理后处理的预设词列表
prepositions = "behind/in/under/on/to/above/left/right/next/up/down/out/of/'s/is".split('/')

def postprocess_text(formatted_text):
    """后处理格式化文本，优化分割结果"""
    formatted_text = formatted_text.replace('| . |', '.')
    formatted_text = formatted_text.replace('| .', '.')
    formatted_text = formatted_text.replace('. |', '.')

    for i in range(len(formatted_text)):
        if i >= len(formatted_text):
            break
        if formatted_text[i] == '|':
            last_dot_index = formatted_text[:i].rfind('.')
            next_space_index = formatted_text[i+2:].find(' ')
            if next_space_index == -1:
                next_space_index = len(formatted_text[i+2:])

            if last_dot_index != -1 and last_dot_index < i - 2:
                previous_words = formatted_text[last_dot_index + 1: i-1].strip().split()
                if len(previous_words) <= 3 and previous_words and previous_words[0] == 'The':
                    formatted_text = formatted_text[:i] + formatted_text[i + 2:]
                    i -= 2
                    continue

            if i+2 < len(formatted_text) and next_space_index < len(formatted_text[i+2:]):
                next_word = formatted_text[i+2:i+2+next_space_index].strip()
                if next_word in prepositions:
                    formatted_text = formatted_text[:i] + formatted_text[i + 2:]
                    i -= 2
                    continue

            if last_dot_index != -1 and last_dot_index < i - 2:
                previous_words = formatted_text[last_dot_index + 1: i-1].strip().split()
                if previous_words and previous_words[0].lower() in prepositions and 'is' not in previous_words and 'are' not in previous_words:
                    formatted_text = formatted_text[:i] + formatted_text[i + 2:]
                    i -= 2

    formatted_text = formatted_text.replace('.', '.|')
    # 分割并过滤掉空字符串
    chunks = [chunk.strip() for chunk in formatted_text.split('|') if chunk.strip()]
    return chunks

def process_dci_dataset(input_path, output_path):
    """处理DCI数据集，将caption分割成chunks"""
    print(f"加载数据集: {input_path}")
    dataset = load_from_disk(input_path)
    
    # 统计变量
    total_chunks = 0
    total_captions = 0
    total_words = 0
    
    def process_example(example):
        nonlocal total_chunks, total_captions, total_words
        
        caption = example["caption"]
        total_captions += 1
        words = word_tokenize(caption)
        total_words += len(words)
        
        processed_text = process_text(caption)
        chunks = postprocess_text(processed_text)
        total_chunks += len(chunks)
        
        return {"image": example["image"], "segmented_caption": chunks}
    
    print("处理数据集...")
    processed_dataset = {}
    
    # 处理每个分割
    for split_name, split_dataset in dataset.items():
        processed_split = split_dataset.map(
            process_example,
            remove_columns=["caption"],
            desc=f"处理 {split_name} 分割"
        )
        processed_dataset[split_name] = processed_split
    
    # 创建新的数据集字典
    new_dataset = DatasetDict(processed_dataset)
    
    # 计算平均值
    avg_chunks_per_caption = total_chunks / total_captions if total_captions > 0 else 0
    avg_words_per_caption = total_words / total_captions if total_captions > 0 else 0
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总标注数: {total_captions}")
    print(f"总chunks数: {total_chunks}")
    print(f"总单词数: {total_words}")
    print(f"平均每个标注的chunks数: {avg_chunks_per_caption:.2f}")
    print(f"平均每个标注的单词数: {avg_words_per_caption:.2f}")
    
    # 保存处理后的数据集
    print(f"保存数据集到: {output_path}")
    new_dataset.save_to_disk(output_path)
    print(f"✅ 处理完成！")

def main():
    # 输入和输出路径
    input_path = "../data/dci.hf"
    output_path = "../data/_sdci_train.hf"
    
    # 处理数据集
    process_dci_dataset(input_path, output_path)

if __name__ == "__main__":
    main() 