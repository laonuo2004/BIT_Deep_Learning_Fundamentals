#| # 实验四：Seq2seq 机器翻译
#|
#| 本次实验的目标是构建一个基于GRU的序列到序列（Seq2seq）模型，来完成简单的英-中机器翻译任务。
#|
#| **学生视角思考**：
#| Seq2seq是处理序列转换问题的经典模型，比如翻译、对话系统等。它由两部分组成：
#| 1. **编码器 (Encoder)**: 读取并理解整个输入句子，将其压缩成一个固定长度的“思想”向量（上下文向量）。
#| 2. **解码器 (Decoder)**: 根据这个“思想”向量，一个词一个词地生成输出句子。
#| 这个任务的挑战在于处理变长的序列数据，以及如何有效地在编码器和解码器之间传递信息。

#| ## 1. 环境准备与库导入
#|
#| 导入所有必需的库，并设置MindSpore的运行环境。

#-
import os
import re
import unicodedata
import random
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import context, Tensor
from mindspore.dataset import GeneratorDataset

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

#| ## 2. 数据准备与预处理
#|
#| ### 2.1 下载并解析数据
#|
#| 我们首先下载英-中平行语料库。

#-
#!wget -N https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/eng-fra.txt -O eng-fra.txt

#| ### 2.2 定义配置和词典类

#| ### 2.2 定义配置和词典类
#|
#| 我们定义一个配置类来管理超参数，并创建一个词典（Vocabulary）类来处理词汇的映射。

#-
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class Config:
    data_path = 'eng-fra.txt'
    vocab_path = 'vocab.json'
    num_epochs = 10
    batch_size = 128
    hidden_size = 256
    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    learning_rate = 0.01

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \\1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

#| ### 2.3 数据过滤与词典构建
#|
#| 为了简化模型，我们只使用长度小于10的简单句，并过滤掉非 "I am", "He is" 等开头的句子。

#-
eng_prefixes = ("i am ", "i m ", "he is", "she is", "they are", "we are", "you are")

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \\
        len(p[1].split(' ')) < MAX_LENGTH and \\
        p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    print("Reading lines...")
    with open('eng-fra.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\\n')
    
    pairs = [[s for s in l.split('\\t')] for l in lines]
    for pair in pairs:
        pair[1] = pair[2]
        del pair[2]
        pair[0] = normalizeString(pair[0])
    with open('eng-fra.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\\n')
    
    pairs = [[s for s in l.split('\\t')] for l in lines]
    for pair in pairs:
        pair[1] = pair[2]
        del pair[2]
        pair[0] = normalizeString(pair[0])

    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")

    input_lang = Vocabulary(lang1)
    output_lang = Vocabulary(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'cmn', False)
print("Random pair:", random.choice(pairs))


#| ### 2.4 数据转换与Dataset创建
#|
#| 将文本句子转换为模型可以处理的Tensor，并创建MindSpore Dataset。

#-
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return indexes

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def create_dataset(pairs, batch_size):
    tensors = [tensorsFromPair(p) for p in pairs]
    
    def generator():
        for pair_tensor in tensors:
            yield pair_tensor[0], len(pair_tensor[0]), pair_tensor[1], len(pair_tensor[1])
    
    dataset = GeneratorDataset(generator, column_names=["input", "input_len", "target", "target_len"])
    
    # Padding
    dataset = dataset.padded_batch(batch_size, 
                                   pad_info={"input": ([MAX_LENGTH + 1], 0), 
                                             "target": ([MAX_LENGTH + 1], 0)})
    return dataset

train_dataset = create_dataset(pairs, Config.batch_size)#| ## 3. 模型构建
#|
#| 我们现在来定义Seq2seq模型的三个核心组件：编码器、解码器和主控模型。
#|
#| **学生视角思考**：
#| - **Encoder**: 它的工作是“阅读”整个英文句子，并将句子的含义压缩到一个`hidden_state`向量中。我使用了GRU，它比LSTM结构更简单，但效果同样强大。
#| - **Decoder**: 它的工作是“写出”中文句子。它会接收Encoder的`hidden_state`作为初始“思想”，然后一个字一个字地生成翻译。在每个时间步，它都会看着Encoder的输出和自己上一步生成的字，来决定下一步要生成什么。
#| - **Seq2Seq**: 这个类像一个“指挥官”，它首先命令Encoder去理解句子，然后将理解的结果（`hidden_state`）传递给Decoder，让Decoder开始生成翻译。

#-
class Encoder(nn.Cell):
    def __init__(self, input_size, hidden_size, embedding_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def construct(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return ops.Zeros()((1, batch_size, self.hidden_size), mindspore.float32)

class Decoder(nn.Cell):
    def __init__(self, hidden_size, output_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.out = nn.Dense(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(axis=1)

    def construct(self, x, hidden):
        output = self.embedding(x)
        output = ops.ReLU()(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq(nn.Cell):
    def __init__(self, encoder, decoder, max_length=MAX_LENGTH):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

    def construct(self, enc_input, dec_input, teacher_forcing_ratio=0.5):
        batch_size = enc_input.shape[1]
        enc_hidden = self.encoder.initHidden(batch_size)
        
        _, enc_hidden = self.encoder(enc_input, enc_hidden)
        
        dec_hidden = enc_hidden
        dec_input_t = dec_input[0]
        
        decoder_outputs = ops.Zeros()((self.max_length, batch_size, self.decoder.out.out_features), mindspore.float32)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, dec_hidden = self.decoder(dec_input_t, dec_hidden)
                decoder_outputs[di] = decoder_output
                dec_input_t = dec_input[di+1] # Teacher forcing
        else:
            for di in range(self.max_length):
                decoder_output, dec_hidden = self.decoder(dec_input_t, dec_hidden)
                topv, topi = decoder_output.topk(1)
                dec_input_t = topi.squeeze().detach()
                decoder_outputs[di] = decoder_output

        return decoder_outputs#| ## 4. 训练与评估
#|
#| 最后，我们定义损失函数、训练逻辑和评估函数，并启动训练。
#|
#| **学生视角思考**：
#| - **Masked Loss**: 我们的输入数据经过了填充（Padding）以对齐长度。在计算损失时，我们必须“忽略”这些填充位，否则模型会去学习拟合这些无意义的填充符。因此，我需要一个带掩码（Mask）的损失函数，它只计算非填充部分的损失。
#| - **Teacher Forcing**: 这是训练Seq2seq模型的一个技巧。在训练初期，我们强制解码器使用真实的上一时间步的输出来预测当前步，而不是它自己生成的（可能错误的）输出。这能让模型更快地收敛。随着训练的进行，我们会逐渐减小Teacher Forcing的比例，让模型学会依赖自己的预测。
#| - **评估**: 训练完成后，我会用几个英文句子来测试模型，看看它能否生成合理的中文翻译。

#-
class MaskedCrossEntropy(nn.Cell):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')

    def construct(self, logits, target, mask):
        mask = mask.astype(mindspore.float32)
        loss = self.criterion(logits, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss

def train_step(input_tensor, target_tensor, seq2seq_model, optimizer, criterion):
    def forward_fn():
        decoder_outputs = seq2seq_model(input_tensor, target_tensor)
        
        mask = ops.ones_like(target_tensor)
        mask[target_tensor == 0] = 0 # Mask padding
        
        loss = criterion(decoder_outputs.view(-1, output_lang.n_words), 
                         target_tensor.view(-1), 
                         mask.view(-1))
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
    loss, grads = grad_fn()
    optimizer(grads)
    return loss

def train_model():
    encoder = Encoder(input_lang.n_words, Config.hidden_size, Config.encoder_embedding_dim)
    decoder = Decoder(Config.hidden_size, output_lang.n_words, Config.decoder_embedding_dim)
    seq2seq_model = Seq2Seq(encoder, decoder)

    optimizer = nn.Adam(seq2seq_model.trainable_params(), learning_rate=Config.learning_rate)
    criterion = MaskedCrossEntropy()

    print("Starting training...")
    for epoch in range(Config.num_epochs):
        total_loss = 0
        for i, (input_tensor, _, target_tensor, _) in enumerate(train_dataset.create_tuple_iterator()):
            loss = train_step(input_tensor, target_tensor, seq2seq_model, optimizer, criterion)
            total_loss += loss.asnumpy()

        print(f'Epoch {epoch + 1}/{Config.num_epochs}, Loss: {total_loss / (i + 1):.4f}')
    
    print("Training complete.")
    return seq2seq_model

def evaluate(seq2seq_model, sentence):
    print("Input:", sentence)
    with mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE):
        input_tensor = Tensor([tensorFromSentence(input_lang, sentence)], mindspore.int32).T
        
        batch_size = input_tensor.shape[1]
        enc_hidden = seq2seq_model.encoder.initHidden(batch_size)
        _, enc_hidden = seq2seq_model.encoder(input_tensor, enc_hidden)
        
        dec_hidden = enc_hidden
        decoder_input = Tensor([[SOS_token]], mindspore.int32)
        
        decoded_words = []
        for _ in range(MAX_LENGTH):
            decoder_output, dec_hidden = seq2seq_model.decoder(decoder_input, dec_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.asnumpy()[0] == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[topi.asnumpy()[0]])
            decoder_input = topi.squeeze().detach()

        print("Output:", ' '.join(decoded_words))

#| ## 5. 执行训练和评估

#-
trained_model = train_model()

#| ### 评估示例

#-
evaluate(trained_model, "i m ok .")
evaluate(trained_model, "he is a reporter .")
evaluate(trained_model, "she is sad .")