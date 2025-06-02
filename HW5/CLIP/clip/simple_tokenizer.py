import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    """
    返回默认的BPE词汇文件路径。
    - 使用lru_cache缓存结果，避免重复计算。
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    创建UTF-8字节到Unicode字符的映射。
    - 解决BPE算法在处理原始UTF-8字节时可能遇到的问题，特别是对于非ASCII字符。
    - 将256个可能的字节映射到Unicode字符，其中ASCII可见字符直接映射，其他字节映射到从2^8开始的Unicode码点。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    从一个单词（由符号元组表示）中获取所有相邻符号对。
    - 用于BPE算法中查找最频繁的字节对。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """
    对文本进行基本清理。
    - 使用ftfy修复文本中的Unicode问题。
    - 解码HTML实体。
    - 移除首尾空白。
    """
    text = ftfy.fix_text(text) # 修复Unicode文本中的常见问题
    text = html.unescape(html.unescape(text)) # 解码HTML实体，两次以处理双重编码
    return text.strip() # 移除字符串两端的空白字符


def whitespace_clean(text):
    """
    清理文本中的多余空白。
    - 将连续的空白字符替换为单个空格。
    - 移除首尾空白。
    """
    text = re.sub(r'\s+', ' ', text) # 将一个或多个空白字符替换为单个空格
    text = text.strip() # 移除字符串两端的空白字符
    return text


class SimpleTokenizer(object):
    """
    CLIP使用的简单分词器。
    - 结合了字节对编码（BPE）和一些文本预处理步骤。
    - 支持编码（文本到token ID）和解码（token ID到文本）。
    """
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode() # 字节到Unicode字符的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # Unicode字符到字节的映射
        
        # 加载BPE合并规则
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1] # 过滤掉文件头和特殊token
        merges = [tuple(merge.split()) for merge in merges] # 将合并规则转换为元组对

        # 构建词汇表
        vocab = list(bytes_to_unicode().values()) # 初始词汇表包含所有Unicode字节映射字符
        vocab = vocab + [v+'</w>' for v in vocab] # 添加表示单词结束的'</w>'后缀
        for merge in merges:
            vocab.append(''.join(merge)) # 添加BPE合并后的新token
        vocab.extend(['<|startoftext|>', '<|endoftext|>']) # 添加特殊token
        
        self.encoder = dict(zip(vocab, range(len(vocab)))) # token到ID的映射
        self.decoder = {v: k for k, v in self.encoder.items()} # ID到token的映射
        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # BPE合并规则的优先级（rank越小优先级越高）
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'} # BPE缓存，避免重复计算
        
        # 正则表达式模式，用于将文本分割成基本token
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        对单个token应用BPE算法。
        - 递归地合并最频繁的字节对，直到无法合并或只剩一个token。
        """
        if token in self.cache:
            return self.cache[token] # 如果已缓存，直接返回
        
        # 将token转换为BPE处理的格式，例如 "hello" -> ("h", "e", "l", "l", "o</w>")
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word) # 获取所有相邻符号对

        if not pairs:
            return token+'</w>' # 如果没有对，直接返回带</w>的token

        while True:
            # 找到优先级最高的（rank最小的）字节对进行合并
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # 如果没有可合并的对，停止
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second) # 合并字节对
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break # 如果只剩一个token，停止
            else:
                pairs = get_pairs(word) # 重新计算符号对

        word = ' '.join(word) # 将合并后的符号连接成字符串
        self.cache[token] = word # 缓存结果
        return word

    def encode(self, text):
        """
        将文本编码为token ID序列。
        - 对文本进行基本清理和空白清理。
        - 使用正则表达式分割文本。
        - 将每个token转换为Unicode表示。
        - 对每个Unicode token应用BPE。
        - 将BPE后的token转换为ID。
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower() # 文本预处理：清理、小写
        for token in re.findall(self.pat, text): # 使用正则表达式分割文本
            # 将token的UTF-8字节转换为Unicode字符表示
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对Unicode token应用BPE，并将BPE后的子token转换为ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        将token ID序列解码回文本。
        - 将token ID转换为BPE token字符串。
        - 将Unicode字符表示转换回原始UTF-8字节。
        - 解码UTF-8字节为字符串。
        - 移除BPE添加的'</w>'标记。
        """
        text = ''.join([self.decoder[token] for token in tokens]) # 将ID转换为BPE token字符串
        # 将Unicode字符表示转换回原始UTF-8字节，然后解码为字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ') # 移除</w>标记
        return text
