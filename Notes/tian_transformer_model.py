# Description: modules for transformer model
import torch 
import torch.nn as nn
import numpy as np
import math
import copy

def clone_module(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def gen_subsequent_mask(seq_len):
    """Mask out subsequent positions. """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0) # (1, seq_len, seq_len)
    return mask 



# Positional Encoding
class VanillaPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len=100, dropout=0.0, dtype=torch.float32):
        super(VanillaPositionalEncoding, self).__init__()
        position = torch.arange(0, max_seq_len, dtype=dtype).unsqueeze(1) # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype) * (-(math.log(10000.0) / d_model))) # (d_model/2, ))
        PETable = torch.zeros(max_seq_len, d_model, dtype=dtype)
        PETable[:, 0::2] = torch.sin(position * div_term)
        PETable[:, 1::2] = torch.cos(position * div_term) # (max_seq_len, d_model)
        self.register_buffer('PETable', PETable.unsqueeze(0)) # (1, max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.PETable[:,:x.shape[1],:].requires_grad_(False) # (1, seq_len, d_model)
        return self.dropout(x)
    
# Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.linears = clone_module(nn.Linear(d_model,d_model), 4)
        self.attn_weights = None      
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_lin = nn.Dropout(dropout)
        
            
    def forward(self, query, key, value, mask=None):
        """
        Inputs:  
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, max_seq_len)
        """
        
        if mask is not None:
            mask = mask.unsqueeze(1) # (batch_size, 1, seq_len, max_seq_len) -> broadcast at num_heads dim        

        batch_size = query.size(0)
        
        # linear projections of query, key, value -> (batch_size, num_heads, seq_len, head_size)
        query, key, value = [
            lin(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1,2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # calculate attention
        x, self.attn_weights = attention(query, key, value, mask=mask, dropout=self.dropout_attn)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)

        # delete intermediate tensors to free up memeory
        del query, key, value
        
        # return output
        x = self.dropout_lin(self.linears[-1](x))
        return x # (batch_size, seq_len, d_model)
        

def attention(query, key, value, mask=None, dropout=None):
     """
     Compute scaled dot product attention
     
        Inputs:
            query, key, value: (batch_size, num_heads, seq_len, head_size)
            mask: (seq_len, seq_len)
     
        Outputs:
            attn_output: (batch_size, num_heads, seq_len, head_size)
            attn_weight: (batch_size, num_heads, seq_len, seq_len)
     """ 
     
     # debug
    #  print("query shape: ", query.shape)
    #  print("mask shape: ", mask.shape)
     
     head_size = query.size(-1)
     scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(head_size) # (batch_size, num_heads, seq_len, seq_len)
     if mask is not None:
         scores = scores.masked_fill(mask==0, float('-inf'))
     attn_weights = nn.functional.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
     if dropout is not None:
         attn_weights = dropout(attn_weights)
     attn_output = torch.matmul(attn_weights, value) # (batch_size, num_heads, seq_len, head_size)
     
     return attn_output, attn_weights

 
# Feed Forward
class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x):
        return self.net(x)


# sublayer connection
class SublayerConnection(nn.Module):
    
    def __init__(self, d_model, dropout=0.0):
        super(SublayerConnection, self).__init__()
        self.ln = nn.LayerNorm(d_model) # layer normalization
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.ln(x)))
        return x
    
# Encoder
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.0):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.self_attn = self_attn
        self.ffn = feed_forward
        self.sublayer_connection = clone_module(SublayerConnection(d_model, dropout), 2)
        
        
    def forward(self, x, mask=None):
                
        # 1) self attention
        x = self.sublayer_connection[0](x, lambda x: self.self_attn(x,x,x, mask))
        # 2) feed forward
        x = self.sublayer_connection[1](x, self.ffn)
        return x
    

class Encoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = clone_module(encoder_layer, num_layers)
        self.ln_final = nn.LayerNorm(encoder_layer.d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return x
    

# Decoder
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = feed_forward
        self.sublayer_connection = clone_module(SublayerConnection(d_model, dropout), 3)
    
    def forward(self, x, encoder_memory, src_mask=None, tgt_mask=None):
        # 1) self attention
        x = self.sublayer_connection[0](x, lambda x: self.self_attn(x,x,x, tgt_mask)) 
        # 2) cross attention (query -> x, key -> encoder_memory, value -> encoder_memory)
        x = self.sublayer_connection[1](x, lambda x: self.cross_attn(x, encoder_memory, encoder_memory, src_mask))
        # 3) feed forward
        x = self.sublayer_connection[2](x, self.ffn)
        return x

class Decoder(nn.Module):
    
    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = clone_module(decoder_layer, num_layers)
        self.ln_final = nn.LayerNorm(decoder_layer.d_model)
    
    def forward(self, x, encoder_memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_memory, src_mask, tgt_mask)
        x = self.ln_final(x)
        return x

# Embedding
class LinearEmbedding(nn.Module):
    
    def __init__(self, d_feature, d_model):
        super(LinearEmbedding, self).__init__()
        self.lut = nn.Linear(d_feature, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Generator: maps the output of the decoder to the target
class Generator(nn.Module):
    
    def __init__(self, d_model, d_feature):
        super(Generator, self).__init__()
        self.lut = nn.Linear(d_model, d_feature)
    
    def forward(self, x):
        return self.lut(x)
              
    
# assembling the model

# 1) full transformer model: includes encoder, decoder, source and target embeddings, and generator
class Full_Transformer_Encoder_Decoder(nn.Module):
    
    def __init__(self, encoder, decoder, src_embedder, tgt_embedder, generator):
        super(Full_Transformer_Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.generator = generator
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_memory = self.encoder(self.src_embedder(src), src_mask)
        decoder_output = self.decoder(self.tgt_embedder(tgt), encoder_memory, src_mask, tgt_mask)
        output = self.generator(decoder_output)
        
        return output

def make_full_transfomer(src_size, tgt_size, d_model, d_ff, num_heads, num_layers, max_seq_len, dropout=0.0):
    attn = MultiHeadAttention(d_model, num_heads, dropout)
    ffn = FeedForward(d_model, d_ff, dropout)
    pe = VanillaPositionalEncoding(d_model, max_seq_len) # dropout is not added here
    
    encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ffn), dropout), num_layers)
    decoder = Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ffn), dropout), num_layers)
    src_embedder = nn.Sequential(LinearEmbedding(src_size, d_model), copy.deepcopy(pe))
    tgt_embedder = nn.Sequential(LinearEmbedding(tgt_size, d_model), copy.deepcopy(pe))
    generator = Generator(d_model, tgt_size)
    
    model = Full_Transformer_Encoder_Decoder(encoder, decoder, src_embedder, tgt_embedder, generator)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        # bias initialization
        else:
            nn.init.constant_(p, 0.0)
    
    return model
    

# 2) decoder only transformer model: includes decoder, source embeddings, and generator
class Decoder_Only_Transformer(nn.Module):
    
    def __init__(self, decoder, embedder, generator):
        super(Decoder_Only_Transformer, self).__init__()
        self.decoder = decoder
        self.embedder = embedder
        self.generator = generator
    
    def forward(self, input, mask=None):
        output = self.decoder(self.embedder(input), mask)
        output = self.generator(output)
        return output
        

def make_decoder_only_transformer(d_feature, d_model, d_ff, num_heads, num_layers, max_seq_len=100, dropout=0.0):
    
    attn = MultiHeadAttention(d_model, num_heads, dropout)
    ffn = FeedForward(d_model, d_ff, dropout)
    pe = VanillaPositionalEncoding(d_model, max_seq_len) # dropout is not added here
    
    embedder = nn.Sequential(LinearEmbedding(d_feature, d_model), copy.deepcopy(pe))
    decoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ffn), dropout), num_layers)
    generator = Generator(d_model, d_feature)
    
    model = Decoder_Only_Transformer(decoder, embedder, generator)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        # bias initialization
        else:
            nn.init.constant_(p, 0.0)
    
    return model

