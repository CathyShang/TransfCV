# Vanilla Transformer
## 位置编码
使用不同频率的sin和cos进行位置编码，公式如下：
$PE_{(pos,i)}=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$
$PE_{(pos,i+1)}=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

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
max_len = 5000, 
## self-attention