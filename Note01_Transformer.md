# Vanilla Transformer
该模型之前有看过有哪几块顺序结构很清楚，但是对于代码层面一直存疑。
- 由于对于解码器网上博客讲述编码器的偏少，所以会重点关注。
- 损失函数，真值输入方式？


## 位置编码
在模型输入过程中，重点理解位置编码过程。词嵌入式过程在其他NLP问题中是通用的。使用不同频率的sin和cos进行位置编码，公式如下：

$PE_{(pos,i)}=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

$PE_{(pos,i+1)}=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

位置编码与词嵌入相加作为模型的输入。

代码实现见下。

## self-attention
在编码器与解码器中均有使用。输入Q(query)、K(key)、V(value),通过下列公式获得注意力的值。

$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

在MultiheadAttention中，将输入词嵌入的维度进行分割，分别输入self-attention层中。代码中使用python中的copy函数。包括编码器/解码器中有多个编码层/解码层，也是使用这样的操作。
```python
import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

## 掩码 mask
掩码 表示 张量，尺寸不定，数值为0，1，自行定义被遮挡或者不遮挡。
在transformer中，掩码主要作用:
- **屏蔽无效padding区域**，encoder中主要第一个作用。以机器翻译为例，一个batch中不同样本输入的长度可能不一样，这时我们设计一个最大句子长度，然后对于空白区域进行padding填充，这些填充的部分对于模型训练时无意义的，因此需要屏蔽掉对应的区域。
- **屏蔽来自未来的信息**， decoder中同时发挥两个作用。Attention机制下，解码的过程中可能会获得未来的信息，这点需要避免。

## 解码器

- 细节:第二层多头注意力机制层 scr-attn
query来自于解码器，key 和 value来自编码器。给出的理解是，利用解码器已经预测出的信息作为query，去编码器提取各种特征，查找相关信息并融合到当前特征中来完成 预测。
**self-attn 注意力机制中Q=K=V, Q K V都是源自同一输入获取，而在scr-attn中 Q!=K=V**

- 细节:第一层多头注意力机制层 self-attn
掩码作用再次强调。解码器准备生成第一个字符时，模型其实已经传入第一个字符去计算损失，但是我们不希望在生成第一个字符时，模型能够利用这个信息，因此我们会将其遮掩。换句话说，**真值在模型还没完全找到答案时已经在模型中**。这里引出一个问题怎么计算损失？

## 模型总览
./transformer_structure.png

## 代码注意
在编程过程中，代码风格模块化，层层调用，复用性良好。编程设计思想值得借鉴。以下列举最小模块：

输入
- PositionalEncoding 类,位置编码
- Embedding 类，词嵌入
  
模型（编码器/解码器）
- attention 函数，注意力机制
- MultiHeadedAttention 类，多头注意力机制
- PositionwiseFeedForward 类，前馈全连接
- LayerNorm 类，规范化层（层归一化）
- subsequent_mask 函数，掩码

输出
- Generator 类，生成器
  
所以想要明白源码意思可以自底向上理解，第一部分就是先理解最小模块。

### 位置编码
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        函数作用:位置编码器类的初始化函数
        
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 下面为模块添加一个持续性缓冲
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 输入 input embedding + PE
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```
### self-attention
```python
def attention(query, key, value, mask = None, dropout = None):
    d_model = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_model)
    # masked_fill 当mask为0，将对应的额scores修改为-1e9
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_atten = F.softmax(scores,dim=-1)
    # 使用dropout随机位置0
    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten
```
### 其他
```python
# 初始化参数
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

## 公式汇总
$PE_{(pos,i)}=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

$PE_{(pos,i+1)}=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

$FFN(x) = max(0,xW_{1}+b_{1})W_{2}+b_{2}$
