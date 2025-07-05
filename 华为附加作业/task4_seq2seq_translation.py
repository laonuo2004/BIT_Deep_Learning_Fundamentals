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
#| 我们首先从国内可访问的镜像下载英-中平行语料库。

#-
#!wget -N http://www.manythings.org/anki/cmn-eng.zip -O cmn-eng.zip
#!unzip -o cmn-eng.zip

#| ### 2.2 定义配置和词典类
#|
#| 我们定义一个配置类来管理超参数，并创建一个词典（Vocabulary）类来处理词汇的映射。

#-
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class Config:
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

#-
eng_prefixes = ("i am ", "i m ", "he is", "she is", "they are", "we are", "you are")

def filterPair(p):
    # Removing the strict prefix filter to accommodate the new dataset.
    return (len(p[0].split(' ')) < MAX_LENGTH and
            len(p[1].split(' ')) < MAX_LENGTH)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2):
    print("Reading lines...")
    
    # 检查文件是否存在
    import os
    if not os.path.exists('cmn.txt'):
        print("错误：找不到 cmn.txt 文件")
        print("请确保 cmn.txt 文件在当前工作目录中")
        print(f"当前工作目录：{os.getcwd()}")
        return None, None, []
    
    with open('cmn.txt', 'r', encoding='utf-8') as f:
        content = f.read().strip()
        print(f"文件内容长度：{len(content)} 字符")
        print(f"文件前100个字符：{repr(content[:100])}")
        
        # 修复：使用正确的转义字符
        lines = content.split('\n')  # 使用 '\n' 而不是 '\\n'

    print(f"共读取 {len(lines)} 行")
    
    # Split every line into pairs, taking only the first two columns (English, Chinese)
    # And only normalize the English sentence
    pairs = []
    for i, l in enumerate(lines):
        if not l.strip():  # 跳过空行
            continue
            
        # 修复：使用正确的转义字符
        parts = l.split('\t')  # 使用 '\t' 而不是 '\\t'
        
        if len(parts) >= 2:
            pairs.append([normalizeString(parts[0]), parts[1]]) # Keep Chinese part as is
        else:
            if i < 5:  # 只打印前5行的调试信息
                print(f"第 {i+1} 行格式不正确，parts数量：{len(parts)}, 内容：{repr(l[:50])}")

    print(f"Read {len(pairs)} sentence pairs")
    
    if len(pairs) == 0:
        print("警告：没有读取到任何有效的句子对！")
        print("请检查文件格式是否正确（应该是制表符分隔的格式）")
        return None, None, []
    
    input_lang = Vocabulary(lang1)
    output_lang = Vocabulary(lang2)

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    print(f"Counted words for {input_lang.name}: {input_lang.n_words}")
    print(f"Counted words for {output_lang.name}: {output_lang.n_words}")
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'cmn')

# 添加安全检查
if pairs and len(pairs) > 0:
    print(f"\nRandom pair: {random.choice(pairs)}")
    print(f"样本总数：{len(pairs)}")
    print(f"第一个样本：{pairs[0]}")
else:
    print("\n错误：没有成功读取到数据，无法继续执行")
    print("请检查以下问题：")
    print("1. cmn.txt 文件是否存在")
    print("2. 文件格式是否正确（制表符分隔）")
    print("3. 文件编码是否为 UTF-8")
    exit(1)


#| ### 2.4 数据转换与Dataset创建

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
    
    dataset = dataset.batch(batch_size,
                            drop_remainder=True,
                            pad_info={"input": ([MAX_LENGTH + 1], 0),
                                      "target": ([MAX_LENGTH + 1], 0)})
    return dataset

train_dataset = create_dataset(pairs, Config.batch_size)

#| ## 3. 模型构建

#-
class Encoder(nn.Cell):
    def __init__(self, input_size, hidden_size, embedding_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def construct(self, x, hidden):
        embedded = self.embedding(x).view(1, x.shape[0], -1)
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
        output = self.embedding(x).view(1, x.shape[0], -1)
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
        
        encoder_outputs = ops.Zeros()((self.max_length, batch_size, self.encoder.hidden_size), mindspore.float32)

        for ei in range(enc_input.shape[0]):
            encoder_output, enc_hidden = self.encoder(enc_input[ei], enc_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        dec_hidden = enc_hidden
        dec_input_t = Tensor(np.array([SOS_token] * batch_size), mindspore.int32)
        
        decoder_outputs = ops.Zeros()((self.max_length, batch_size, self.decoder.out.out_features), mindspore.float32)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, dec_hidden = self.decoder(dec_input_t, dec_hidden)
                decoder_outputs[di] = decoder_output
                dec_input_t = dec_input[di]
        else:
            for di in range(self.max_length):
                decoder_output, dec_hidden = self.decoder(dec_input_t, dec_hidden)
                topv, topi = decoder_output.topk(1)
                dec_input_t = topi.squeeze().detach()
                decoder_outputs[di] = decoder_output

        return decoder_outputs

#| ## 4. 训练与评估

#-
class MaskedNLLLoss(nn.Cell):
    def __init__(self):
        super(MaskedNLLLoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='none')
    
    def construct(self, inp, target, mask):
        mask = mask.astype(mindspore.float32)
        loss = self.loss(inp, target) * mask
        return loss.sum() / mask.sum()

def train_step(input_tensor, target_tensor, seq2seq, optimizer, criterion):
    def forward_fn():
        decoder_outputs = seq2seq(input_tensor, target_tensor)
        mask = ops.ones_like(target_tensor)
        mask[target_tensor == 0] = 0
        loss = criterion(decoder_outputs.view(-1, output_lang.n_words), 
                         target_tensor.view(-1), 
                         mask.view(-1))
        return loss
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
    loss, grads = grad_fn()
    optimizer(grads)
    return loss.asnumpy()

def train_model():
    encoder = Encoder(input_lang.n_words, Config.hidden_size, Config.encoder_embedding_dim)
    decoder = Decoder(Config.hidden_size, output_lang.n_words, Config.decoder_embedding_dim)
    seq2seq_model = Seq2Seq(encoder, decoder)

    optimizer = nn.Adam(seq2seq_model.trainable_params(), learning_rate=Config.learning_rate)
    criterion = MaskedNLLLoss()

    print("Starting training...")
    for epoch in range(Config.num_epochs):
        total_loss = 0
        for i, (inp, inp_len, tar, tar_len) in enumerate(train_dataset.create_tuple_iterator()):
            loss = train_step(inp.T, tar.T, seq2seq_model, optimizer, criterion)
            total_loss += loss
        print(f'Epoch {epoch + 1}/{Config.num_epochs}, Loss: {total_loss / (i + 1):.4f}')
    
    print("Training complete.")
    return seq2seq_model

def evaluate(seq2seq_model, sentence):
    print("Input:", sentence)
    with mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE):
        input_tensor = Tensor([tensorFromSentence(input_lang, sentence)], mindspore.int32).T
        
        batch_size = input_tensor.shape[1]
        enc_hidden = seq2seq_model.encoder.initHidden(batch_size)

        for ei in range(input_tensor.shape[0]):
            _, enc_hidden = seq2seq_model.encoder(input_tensor[ei], enc_hidden)
        
        dec_hidden = enc_hidden
        decoder_input = Tensor([[SOS_token]], mindspore.int32)
        
        decoded_words = []
        for _ in range(MAX_LENGTH):
            decoder_output, dec_hidden = seq2seq_model.decoder(decoder_input, dec_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.asnumpy()[0] == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[topi.asnumpy()[0][0]])
            decoder_input = topi.squeeze().detach().view(1, -1)

        print("Output:", ' '.join(decoded_words))

#| ## 5. 执行训练和评估

#-
trained_model = train_model()

#| ### 评估示例

#-
evaluate(trained_model, "i m ok .")
evaluate(trained_model, "he is a reporter .")
evaluate(trained_model, "she is sad .")