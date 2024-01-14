import torch
from torch import Tensor 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def attention(query:Tensor, key:Tensor, value:Tensor):
    """ 
    计算Attention的结果。
    这里其实传入对的是Q，K，V；而Q，K，V的计算是放在模型中的，请参考后续的MultiHeadAttention类。
    
    这里的Q，K，V有两种shape，如果是Self-Attention，shape为(batch, 词数, d_model),
                            例如(1, 7, 128)，表示batch_size为1，一句7个单词，每个单词128维
                            
                            但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
                            例如(1, 8, 7, 16)，表示batch_size为1,8个head，一句7个单词，128/8=16。
                            
                            这样其实也能看出来，所谓的MultiHead其实也就是将128拆开了。
                            
                            在Transformer中，由于使用的是MultiHead Attention，所以Q、K、V的shape只会是第二种。
    """

    """ 
    获取 d_model 的值。之所以这样可以获取，是因为 query 和输入的 shape 相同。
    若为Self-Attention，则最后一维都是词向量的维度，也就是 d_model 的值；
    若为MultiHead-Attention，则最后一维是 d_model/h，h表示head数。
    """
    d_k = query.size(-1)
    
    # 执行QK^T / 根号下d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    """ 
    执行公式中的softmax
    这里的 p_attn 是一个方阵；若为Self-Attention，则shape为(batch, 词数， 词数)；
    若为MultiHead-Attention，则shape为(batch, head数, 词数, 词数)
    """
    p_attn = scores.softmax(dim=-1)
    
    """ 
    最后再乘以 V.
    对于Self-Attention来说，结果 shape 为(batch, 词数, d_model)，这也就是最终的结果了。
    对于MultiHead-Attention来说，结果 shape 为(batch, head数, 词数, d_model/head数)
    而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention该做的事。
    """
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, h:int, d_model:int) -> None:
        """ 
        h: head数
        d_model: d_model数
        """
        super().__init__()
        
        assert d_model % h == 0, "head number should be divided by d_model"
        
        self.d_k = d_model // h
        self.h = h

        # 定义W^q、W^k、W^v和W^o矩阵。
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model)
        ]
    
    def forward(self, x):
        # 获取batch_size
        batch_size = x.size(0)
        
        """ 
        1. 求出Q、K、V。这里是求MultiHead的Q、K、V，所以shape为(batch, head数, 词数, d_model/head数)
            1.1 首先，通过定义的W^q, W^k, W^v 求出Self-Attention的Q、K、V。此时，Q、K、V的shape为(batch, 词数, d_model)
                对应代码为 linear(x)
            1.2 分为多头，即将shape由(batch, 词数, d_model)变为(batch, 词数, head数, d_model/head数)
                对应代码为 .view(batch_size, -1, self.h, self.d_k)
            1.3 最终交换 词数 和 head数 这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数, d_model/head数)
                对应代码为 .transpose(1,2)
        """
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:-1], (x, x, x))]

        """ 
        2. 求出Q、K、V后，通过Attention函数计算出Attention结果。
            这里x的shape为(batch, head数, 词数, d_model/head数)
            self.attn的shape为(batch, head数, 词数, 词数)
        """
        x = attention(query, key, value)
        
        """ 
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数, d_model/head数)再变为(batch, 词数, d_model)
            3.1 首先, 交换 head数 和 词数 维度，结果为 (batch, 词数, head数, d_model/head数)
                对应代码为
        """
        x = x.transpose(1,2).reshape(batch_size, -1, self.h * self.d_k)

        """ 
        4. 最后，通过W^o矩阵再执行一次线性变换，得到最终结果
        """
        return self.linears[-1](x)
