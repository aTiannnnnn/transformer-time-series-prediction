import os
# Get the path to the current file
current_file_path = os.path.dirname(__file__)
os.chdir(current_file_path)

import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import copy
from dataclasses import dataclass
from tian_transformer_model import *
from tian_transformer_training import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

#######################################################
# Parameters Definition
#######################################################
@dataclass
class data_params:
    train_ratio: float = 0.8
    input_window: int = 100
    output_window: int = 1
    batch_size: int = 10
    mask: torch.Tensor = gen_subsequent_mask(100)

@dataclass
class model_params:
    d_feature: int = 1
    d_model: int = 250
    d_ff: int = 1000
    num_heads: int = 10
    num_layers: int = 1
    max_seq_len: int = 100
    dropout: float = 0.1


train_data, val_data = get_temperature_data(data_params.input_window, data_params.output_window, data_params.train_ratio) # (num_samples, 2, input_window) including input and label
model = make_decoder_only_transformer(d_feature=model_params.d_feature, 
                                      d_model=model_params.d_model, 
                                      d_ff=model_params.d_ff, 
                                      num_heads=model_params.num_heads, 
                                      num_layers=model_params.num_layers, 
                                      max_seq_len=model_params.max_seq_len, 
                                      dropout=model_params.dropout)
model.to(device)
get_num_params(model)

criterion = nn.MSELoss()
lr = 0.001 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
best_val_loss = float("inf")
epochs = 100 # The number of epochs
best_model = None


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss = train_epoch(train_data, data_params, model, criterion, optimizer, scheduler, epoch)
    if ( epoch % 5 == 0 ):
        val_loss = one_step_prediction(val_data, data_params, model, criterion, epoch)
        pred_loss = autogressive_prediction(val_data, data_params, model, criterion, 600, epoch)
    else:
        val_loss = evaluate(val_data, data_params, model, criterion)
   
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     train_loss, val_loss, math.exp(val_loss)))
    print('-' * 89)




