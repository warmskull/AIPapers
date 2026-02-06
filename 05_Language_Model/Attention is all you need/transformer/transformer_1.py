import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

''' mini_batch迭代器 '''
def make_batch(sentences):
    '''
    sentence = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    '''
    input_batch = [[src_vocab[n] for n in sentences[0].split()]] # [[1, 2, 3, 4, 0]]    输入数据集
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]] # [[5, 1, 2, 3, 4]]   输出数据集
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]] # [[1, 2, 3, 4, 6]]   目标数据集
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)
# enc_inputs, dec_inputs, target_batch = make_batch(sentences)
# enc_inputs = tensor([[1, 2, 3, 4, 0]])
# dec_inputs = tensor([[5, 1, 2, 3, 4]])
# target_batch = tensor([[1, 2, 3, 4, 6]])


'''==================================两种mask===================================='''

''' 
Padding Mask形成一个符号矩阵   
padding mask的作用：不同batch之间句子长度可以不一样，但是每个batch的长度必须是一样的：
因此出现一个问题，不够长度需要加pad，使得其长度变成一样。
# enc_inputs = tensor([[1, 2, 3, 4, 0]])    dec_inputs = tensor([[5, 1, 2, 3, 4]])
'''
def get_attn_pad_mask(seq_q, seq_k):   # seq_q只是给个维度 互自相关时候 横轴和竖轴可能不同
    batch_size, len_q = seq_q.size() # batch_size=1   len_q=5
    batch_size, len_k = seq_k.size() # batch_size=1   len_k=5
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    '''
seq_k.data.eq(0)，
这句的作用是返回一个大小和seq_k一样的 tensor，只不过里面的值只有 True 和 False。
如果 seq_k 某个位置的值等于 0，那么对应位置就是True，否则即为False。
举个例子，输入为 seq_data = [1, 2, 3, 4, 0]，
seq_data.data.eq(0)就会返回[False, False, False, False, True]
    '''
    return pad_attn_mask.expand(batch_size, len_q, len_k)
''' 因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量

enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) = enc_inputs自注意力层的时候PAD部分
        tensor([[[False, False, False, False,  True],
                [False, False, False, False,  True],     横轴enc_inputs(ich mochte ein bier P) 和 竖轴enc_outputs(ich mochte ein bier P)
                [False, False, False, False,  True],     竖轴经注意力机制后生成得ich是带有原ich mochte ein bier P关系得ich 它和原P没关系 原P仅仅是填充
                [False, False, False, False,  True],     最后一行P也是[False, False, False, False,  True]因为 这个新得填充P也要和原语句建立联系
                [False, False, False, False,  True]]])         当对每一行做Softmax时候，我们对True这里做无穷小  得到结果为0  

dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) = 交互注意力机制PAD部分   其实这里应该是dec_inputs, enc_outputs  只不过不关注PAD用inputs足够了
        tensor([[[False, False, False, False,  True],                                 enc_outputs是生成阿巴阿巴 最后一个位置不是P 是P的新表现形式    中间这些都是未知的
                [False, False, False, False,  True],
                [False, False, False, False,  True],
                [False, False, False, False,  True],
                [False, False, False, False,  True]]])
                
dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) =  dec_inputs自注意力层的时候PAD部分
        tensor([[[False, False, False, False, False],
                [False, False, False, False, False],              横轴dec_inputs(S i want a beer) 和 竖轴 dec_inputs(S i want a beer)
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]])            
'''








'''Sequence Mask屏蔽子序列的mask '''
def get_attn_subsequent_mask(seq):
    """ seq: [batch_size, tgt_len] """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
'''
dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) = 
屏蔽子序列的mask部分，这个函数就是用来表示Decoder的输入中哪些是未来词，使用一个上三角为1的矩阵遮蔽未来词，让当前词看不到未来词
tensor([[[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]], dtype=torch.uint8)
'''






''' 位置编码（Position Embedding） '''
'''
Transformer 中需要使用Position Embedding 表示单词出现在句子中的位置。
因为 Transformer 不采用 RNN 的结构，而是使用全局信息，因此是无法捕捉到序列顺序信息的。
例如将K、V按行进行打乱，那么Attention之后的结果是一样的。但是序列信息非常重要，代表着全局的结构，因此必须将序列的分词相对或者绝对position信息利用起来。
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 共有项，利用指数函数e和对数函数log取下来，方便计算

        pe[:, 0::2] = torch.sin(position * div_term)
        # 同理，这里是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 这里的self.pe是从缓冲区里拿的
        # 切片操作，把pe第一维的前seq_len个tensor和x相加，其他维度不变
        # 实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)







'''Scaled DotProduct Attention缩放点积注意力机制'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # matmul操作即矩阵相乘
        # Q*K转置/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        '''
        拿一头出来看看
          [ 4.4513e-01,  6.0761e-01,  1.3474e+00,  1.2734e+00,  6.3869e-01],
          [-1.2455e-01,  8.5702e-01, -4.0142e-01,  5.8216e-01,  3.9691e-01],
          [ 4.1076e-01,  5.8057e-01,  5.4437e-01,  1.2025e+00,  8.1216e-01],
          [ 5.5553e-01,  1.0125e+00,  3.5327e-01,  9.4872e-02,  5.5170e-01],
          [ 1.1529e-01,  9.6572e-01,  4.1690e-01,  3.2786e-01, -3.6752e-01]],
        '''
        # 把被mask的地方置为无穷小，softmax之后会趋近于0，Q会忽视这部分的权重
        scores.masked_fill_(attn_mask, -1e9)  # 把我们之前设置的mask矩阵中True的地方设置为负无穷
        '''
          [ 4.4513e-01,  6.0761e-01,  1.3474e+00,  1.2734e+00, -1.0000e+09],
          [-1.2455e-01,  8.5702e-01, -4.0142e-01,  5.8216e-01, -1.0000e+09],
          [ 4.1076e-01,  5.8057e-01,  5.4437e-01,  1.2025e+00, -1.0000e+09],
          [ 5.5553e-01,  1.0125e+00,  3.5327e-01,  9.4872e-02, -1.0000e+09],
          [ 1.1529e-01,  9.6572e-01,  4.1690e-01,  3.2786e-01, -1.0000e+09]],
        '''
        # .masked_fill_(mask，value) 是张量操作函数，用于对张量中的部分元素进行替换操作
        # mask：一个与原张量形状相同的布尔类型的张量，用于指示要替换的元素的位置。True 表示需要替换，False 表示不需要替换
        # value：一个标量或大小与原张量相同的张量，用于指定替换后的值
        ''' attn = (1,8,5,5)  V = (1,8,5,64) '''
        attn = nn.Softmax(dim=-1)(scores)  #q求出对应attention score
        context = torch.matmul(attn, V)  # attention*V 新的Z (1,8,5,64)
        return context, attn








'''MultiHead Attention：多头注意力机制'''
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Wq,Wk,Wv其实就是一个线性层，用来将输入映射为Q、K、V
        # 这里输出是d_k * n_heads，因为是先映射，后分头。  d_k=d_v=64   n_heads=8 d_model=512
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    """
        自注意力时，Q,K,V均来自输入带位置信息的词向量，自注意力Q=K=V
        Q,K,V = enc_inputs(1,5,512)  attn_mask = enc_self_attn_mask(1,5,5)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        交叉注意力式，K,V来自enc_outputs  Q来自掩码自注意力后的dec_inputs
    """
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        # 分头；一定要注意的是自注意力时q和k分头之后维度是一致的，所以一看这里都是d_k  转置是因为先分头 再得W_Q
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # (1,8,5,64)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # (1,8,5,64)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) # (1,8,5,64)

        '''mask也进行分头操作'''
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask(1,8,5,5) 就是把pad信息复制n份一样的，重复到n个头上以便计算多头注意力机制
        '''求出新的词向量Z'''
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)  # context(1,8,5,64)  attn(1,8,5,5)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # 转换格式context(1,5,512) 新的词向量Z
        output = self.linear(context) # output(1,5,512)

        return self.layer_norm(output + residual), attn # 残差＋归一化






'''前馈神经网络（PoswiseFeedForward）'''
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),  # 512-->2048
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))  # 2048-->512
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):  # (1,5,512)
        residual = inputs
        output = self.fc(inputs) # (1,5,512)
        return self.layer_norm(output + residual)  # 残差+归一化




# ---------------------------------------------------#
# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
# ---------------------------------------------------#
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """下面这个就是做自注意力层，输入是enc_inputs是带位置编码的了(1,5,512)    enc_self_attn_mask(1,5,5) """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_outputs(1,5,512)  attn(1,8,5,5)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs(1,5,512) 前馈神经网络之后
        return enc_outputs, attn

# -----------------------------------------------------------------------------#
# Encoder部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model) # 这行其实就是生成一个矩阵，算对每个单词编码 src_vocab_size,d_model = (5,512) 每个单词变成512维向量
        self.pos_emb = PositionalEncoding(d_model) # 位置编码，这里是固定的正余弦函数
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) # 使用ModuleList对多个encoder进行堆叠

    def forward(self, enc_inputs):
        # enc_inputs = (1,5)  tensor([[1, 2, 3, 4, 0]])
        enc_outputs = self.src_emb(enc_inputs) # 词向量编码enc_outputs = (1,5,512)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # 包含位置编码词向量 enc_outputs = (1,5,512)
        ''' 现在已经编码好了 encoder带位置信息-->enc_outputs '''
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # 原enc_inputs哪些值是PAD  enc_self_attn_mask = (1,5,5)    get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响  告诉计算机哪一个词使用PAD填充的
        enc_self_attns = []
        for layer in self.layers:
            # 送入6个mulit_head_attention
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



# -----------------------------------------------------------------------------#
# Decoder Layer包含了三个部分：解码器自注意力、“编码器-解码器”注意力、基于位置的前馈网络
# -----------------------------------------------------------------------------#
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''先dec_inputs掩码自注意力机制  使用的是dec_self_attn_mask'''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        '''再dec_outputs与enc_outputs互注意力机制  使用的是dec_enc_attn_mask 看不到后面的+看不到K的PAD'''
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

# -----------------------------------------------------------------------------#
# Decoder 部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs(1,5) enc_inputs(1,5) enc_outputs(1,5,512)
        dec_outputs = self.tgt_emb(dec_inputs)  # 词向量(1,5,512)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # 带位置信息词向量(1,5,512)

        ## get_attn_pad_mask dec_inputs自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) #(1,5,5) T,F
        '''       dec_inputs, dec_inputs
        tensor([[[False, False, False, False, False],
                 [False, False, False, False, False],
                 [False, False, False, False, False],
                 [False, False, False, False, False],
                 [False, False, False, False, False]]])
        '''

        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) # (1,5,5) 0,1
        '''            掩码
        tensor([[[0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        '''

        ## 两个矩阵相加，大于0的为1，不大于0的为0(不是pad且没被掩码)，为1(是pad但掩码1+0，是pad且没掩码1+1)的在之后就会被fill到无限小  我们只喜欢False
        # dec_self_attn_pad_mask + dec_self_attn_subsequent_mask    dec_inputs自注意力层的时候的pad部分 + dec_inputs自注意层的mask部分
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # (1,5,5)
        '''                  dec_inputs, dec_inputs + 掩码
        tensor([[[False,  True,  True,  True,  True],
                 [False, False,  True,  True,  True],
                 [False, False, False,  True,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False, False]]])
        '''

        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是K,V dec的输入是Q  这里只关心k，不关心Q了
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # (1,5,5)
        '''                         dec_inputs, enc_inputs
        tensor([[[False, False, False, False,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False,  True]]])
        '''

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns







class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # 编码层
        self.decoder = Decoder()  # 解码层

        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    # 实现函数
    def forward(self, enc_inputs, dec_inputs):
        """
         Transformers的输入：两个序列（编码端的输入，解码端的输入）
         enc_inputs: [batch_size, src_len]   tensor([[5, 1, 2, 3, 4]]) (1,5)
         dec_inputs: [batch_size, tgt_len]   tensor([[5, 1, 2, 3, 4]]) (1,5)
         """
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]  每个单词都编码成512维且包含位置信息
        enc_outputs, enc_self_attns = self.encoder(enc_inputs) # enc_outputs(1,5,512)  enc_self_attns 6个(1,8,5,5)

        # 经过Decoder网络后，得到的输出还是[batch_size, src_len, d_model]  每个单词都编码成512维且包含位置信息
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs) # dec_outputs(1,5,512)  dec_self_attns6个(1,8,5,5)  dec_enc_attns6个(1,8,5,5)

        # (1,5,512)-->(1,5,7)
        dec_logits = self.projection(dec_outputs) # (1,5,7)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns





''' ===1.训练集（句子输入部分）=== '''
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
''' ===2.测试集（构建词表）=== '''
# 编码端的词表
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_vocab_size = len(src_vocab)  # src_vocab_size：实际情况下，它的长度应该是所有德语单词的个数  5
# 解码端的词表
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
tgt_vocab_size = len(tgt_vocab)  # 实际情况下，它应该是所有英语单词个数  7
# P：pad填充字符。如果当前批次的数据量小于时间步数，将填写空白序列的符号。
# S：Start。显示 解码 输入开始 的符号
# E：End。显示 解码 输出开始 的符号


''' 超参数设置 '''
src_len = 5  # 编码端的输入长度
tgt_len = 5  # 解码端的输入长度

d_model = 512  # 每一个字符转化成Embedding的大小
d_ff = 2048  # 前馈神经网络映射到多少维度 全连接

d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  #   encoder和decoder的个数，这个设置的是6个encoder和decoder堆叠在一起
n_heads = 8  # 多头注意力机制时，把头分为几个，这里说的是分为8个


model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  # 用Adam的话效果不好

enc_inputs, dec_inputs, target_batch = make_batch(sentences)
# enc_inputs = tensor([[1, 2, 3, 4, 0]])
# dec_inputs = tensor([[5, 1, 2, 3, 4]])
# target_batch = tensor([[1, 2, 3, 4, 6]])

for epoch in range(100):
    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs) #outputs(5,7)  enc_self_attns6个(1,8,5,5)   dec_self_attns6个(1,8,5,5)   dec_enc_attns6个(1,8,5,5)
    loss = criterion(outputs, target_batch.contiguous().view(-1))  # 损失函数是 输出的预测 和 真实
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    print(outputs)
    print(target_batch)
    print(target_batch.contiguous().view(-1))
    loss.backward()
    optimizer.step()


