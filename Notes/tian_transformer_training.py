import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#######################################################
# Part 1: Data Preprocessing Functions
#######################################################
def create_inout_sequences(input_data, input_window ,output_window):
    inout_seq = []
    L = len(input_data)
    block_len = input_window + output_window
    block_num =  L - block_len + 1
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window

    for i in range(block_num):
        train_seq = input_data[i : i + input_window] # (input_window,)
        train_label = input_data[i + output_window : i + input_window + output_window] # (input_window,)
        inout_seq.append((train_seq ,train_label))

    return torch.FloatTensor(np.array(inout_seq))

 
def get_temperature_data(input_window=100, output_window=1, train_ratio=0.8):
    
    from sklearn.preprocessing import MinMaxScaler
    from pandas import read_csv
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    csv_data = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    series = csv_data.values[:]
    amplitude = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)
    sampels = int(len(amplitude) * train_ratio)
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]
    
    train_data = create_inout_sequences(train_data,input_window ,output_window)
    test_data = create_inout_sequences(test_data,input_window,output_window)
    
    return train_data.to(device),test_data.to(device)


def get_batch(input_data, offset , batch_size, input_window):

    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(input_data) - offset)
    data = input_data[offset : offset + batch_len]
    input = torch.stack([item[0] for item in data]).view((input_window,batch_len,1))
    # ( seq_len, batch, 1 ) , 1 is feature size
    target = torch.stack([item[1] for item in data]).view((input_window,batch_len,1))
    
    input = input.transpose(0,1)
    target = target.transpose(0,1) # (batch, seq_len, 1)
    
    return input, target




#######################################################
# Part 2: Model Training Functions
#######################################################
def get_num_params(model):
    num_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_paras:,} trainable parameters')


def train_epoch(train_data, 
                data_params,
                model,
                criterion, 
                optimizer, 
                scheduler, 
                curren_epoch):
    
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()
    total_loss_all_batches = 0
    
    batch_size = data_params.batch_size
    input_window = data_params.input_window
    input_mask = data_params.mask
    
    
    for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        
        data, targets = get_batch(train_data, i , batch_size, input_window)
        optimizer.zero_grad()
        output = model(data, input_mask)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        total_loss_all_batches += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    curren_epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
    return total_loss_all_batches / (batch + 1)

def evaluate(val_data, data_params, model,criterion):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    input_mask = data_params.mask
    input_window = data_params.input_window
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1, eval_batch_size): # Now len-1 is not necessary
        for i in range(0, len(val_data), eval_batch_size):
            data, targets = get_batch(val_data, i,eval_batch_size, input_window)
            output = model(data, input_mask)            
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(val_data)



def one_step_prediction(val_data, data_params, model, criterion, epoch):
    
    input_window = data_params.input_window
    input_mask = data_params.mask
    
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1):
        for i in range(len(val_data)):  # Now len-1 is not necessary
            data, target = get_batch(val_data, i , 1, input_window) # one-step forecast
            output = model(data, input_mask)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[:,-1,:].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[:,-1,:].view(-1).cpu()), 0)
    val_loss = total_loss / i
    plt.plot(test_result,color="red")
    plt.plot(truth,color="blue")
    plt.title(f"epoch {epoch}  validation loss: {val_loss:.5f}")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.savefig('graph_new_one_step_preds/transformer-epoch-%03d.png'%epoch)
    plt.close()
    return val_loss


def autogressive_prediction(val_data, data_params, model, criterion, steps, epoch):
    
    input_window = data_params.input_window
    input_mask = data_params.mask
    model.eval() 
    data, _ = get_batch(val_data , 0 , 1, input_window) # (1, input_window, 1)
    truth = copy.deepcopy(data.cpu().view(-1))
    with torch.no_grad():
        for i in range(0, steps):    
            _, target = get_batch(val_data, i , 1, input_window) # one-step forecast
            truth = torch.cat((truth, target[:,-1,:].view(-1).cpu()), 0)        
            output = model(data[:,-input_window:,:], input_mask)
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            
            # debug
            # print(f"data.shape: {data.shape}")
            # print(f"output.shape: {output.shape}")
            # print(f"output[:,-1:,:].shape: {output[:,-1:,:].shape}")
            # -----
            data = torch.cat((data, output[:,-1:,:]), dim=1) # [m,m+1,..., m+n+1]

    data = data.cpu().view(-1)
    pred_loss = criterion(data, truth)
    
    plt.plot(data,color="red")       
    plt.plot(truth,color="blue", linestyle=':')    
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.title(f"epoch {epoch}  validation loss: {pred_loss:.5f}")
    # save the plot
    plt.savefig('graph_new_ag_preds/transformer-epoch-%03d.png'%epoch)
    plt.show()
    plt.close()
    
    return pred_loss



