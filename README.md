手撕Transformer，应用于CV方向。
# Scratch Transformer model 
## 位置编码
使用不同频率的sin和cos进行位置编码，公式如下：
$PE_{(pos,i)}=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$
$PE_{(pos,i+1)}=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$

## self-attention