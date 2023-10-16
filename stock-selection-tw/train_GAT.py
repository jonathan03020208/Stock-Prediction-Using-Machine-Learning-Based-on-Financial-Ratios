### Before running, plz setup the feild [GAT] in config.ini
### python train_GAT.py 2019-3

import random
import pickle
from tqdm import tqdm 
import statistics
import sys
import os
import configparser
import json
import pandas as pd

from GAT import GAT
import torch.nn as nn
import torch

config = configparser.ConfigParser()    
config.read('config.ini')

#device
if config['GAT']['device'].lower() == 'cpu':
    device = torch.device("cpu")
elif config['GAT']['device'].lower() == 'gpu':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    raise RuntimeError('device must be cpu or gpu')
    
#ifDummy
if config['GAT']['dummy'].lower() == 'true':
    ifDummy = True
elif config['GAT']['dummy'].lower() == 'false':
    ifDummy = False
else:
    raise RuntimeError('dummy must be true or false')

#ifRelative
if config['GAT']['relative'].lower() == 'true':
    ifRelative = True
elif config['GAT']['relative'].lower() == 'false':
    ifRelative = False
else:
    raise RuntimeError('relative must be true or false')

if config['GAT']['GAT_cat'].lower() == '1':
    GAT_cat = 1
elif config['GAT']['GAT_cat'].lower() == '2':
    GAT_cat = 2
elif config['GAT']['GAT_cat'].lower() == '3':
    GAT_cat = 3
elif config['GAT']['GAT_cat'].lower() == '4':
    GAT_cat = 4
else:
    raise RuntimeError('GAT_cat must be 1~4')

#dir root
root = config['GAT']['model_path_root'] + '/' + sys.argv[1] + '/'

NUMBER_OF_COMPANY = 97

year = int(sys.argv[1].split('-')[0])
quarter = int(sys.argv[1].split('-')[1])

# 2013-4 => #1
simulation_num = (year - 2013) * 4 + quarter - 3

sector_dict = {
    'Technology': 0,
    'Energy': 1,
    'Consumer Cyclical': 2,
    'Industrials': 3,
    'Consumer Defensive': 4,
    'Real Estate': 5,
    'Financial Services': 6,
    'Communication Services': 7,
    'Basic Materials': 8
}

input = pd.read_csv('ratio-2022-12-13.csv', dtype='str', keep_default_na=False)

number_of_quarters_per_stock = 37
sector_list = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    sector_list.append(sector_dict[input['Sector'][i]])

ratio_raw = input.drop(labels='CO_ID', axis=1)
ratio_raw = ratio_raw.drop(labels='Name', axis=1)
ratio_raw = ratio_raw.drop(labels='Year', axis=1)
ratio_raw = ratio_raw.drop(labels='Quarter', axis=1)
ratio_raw = ratio_raw.drop(labels='Sector', axis=1)
ratio_raw = ratio_raw.drop(labels='Stock Return', axis=1)

ratio_raw = ratio_raw.values.tolist()

ratio = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    ratio_for_one_company_in_right_order = []
    for j in range(i, i + number_of_quarters_per_stock):
        ratio_for_one_company_in_right_order.append(ratio_raw[j])
    ratio.append(ratio_for_one_company_in_right_order)

#determine scope of feature
ratio_scope = []
for i in range(NUMBER_OF_COMPANY):
    ratio_scope.append(ratio[i][simulation_num-24:simulation_num+3])

#補空缺資料:若至少有一個資料點,則補平均,若無則補-1
def isMissing(input):
    return input == '' or input == 'nan'

NumberOfFeatures = 18

for i in range(NUMBER_OF_COMPANY):     
    for k in range(NumberOfFeatures):
        nonMissingCount = 0
        sum = 0.0
        for j in range(len(ratio_scope[i])):
            if not isMissing(ratio_scope[i][j][k]):
                ratio_scope[i][j][k] = float(ratio_scope[i][j][k].replace(',', ''))
                sum += ratio_scope[i][j][k]
                nonMissingCount += 1
        if nonMissingCount:
            mean = sum / nonMissingCount
            for j in range(len(ratio_scope[i])):
                if isMissing(ratio_scope[i][j][k]):
                    ratio_scope[i][j][k] = mean
        else:
            for j in range(len(ratio_scope[i])):
                ratio_scope[i][j][k] = -1.0

#standardization
mean = []
stdev = []
for k in range(NumberOfFeatures):
    arr = []
    for i in range(NUMBER_OF_COMPANY):
        for j in range(len(ratio_scope[i])):
            arr.append(ratio_scope[i][j][k])
    mean.append(statistics.mean(arr))        
    stdev.append(statistics.stdev(arr))

for i in range(NUMBER_OF_COMPANY):
    for j in range(len(ratio_scope[i])):
        for k in range(NumberOfFeatures):
            ratio_scope[i][j][k] = (ratio_scope[i][j][k] - mean[k]) / stdev[k]

if ifDummy:
    for i in range(NUMBER_OF_COMPANY):
        dummy_vector = [0 for _ in range(len(sector_dict) - 1)]
        if sector_list[i] != 8:
            dummy_vector[sector_list[i]] = 1
        for j in range(len(ratio_scope[i])):
            ratio_scope[i][j] = ratio_scope[i][j] + dummy_vector

return_raw = input['Stock Return'].astype(float).values.tolist()

stock_return = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    stock_return.append(return_raw[i+simulation_num-21 : i + number_of_quarters_per_stock])

tw_return = return_raw[-number_of_quarters_per_stock+simulation_num-21:]

stock_relative_return = []
for i in range(NUMBER_OF_COMPANY):
    stock_relative_return.append([stock_return[i][k] - tw_return[k] for k in range(len(stock_return[i]))])

data_X = [] # in shape [quarter_num, stock_num, time_step, ratio]
data_Y = []
for quarter_num in range(1, 24):
    quarter_X_temp = []
    quarter_Y_temp = []
    for stock_num in range(NUMBER_OF_COMPANY):
        quarter_X_temp.append(ratio_scope[stock_num][quarter_num - 1:quarter_num + 3][:])
        if ifRelative:
            quarter_Y_temp.append(stock_relative_return[stock_num][quarter_num - 1])
        else:
            quarter_Y_temp.append(tw_return[stock_num][quarter_num - 1])
    data_X.append(quarter_X_temp)
    data_Y.append(quarter_Y_temp)

data_X = torch.Tensor(data_X)
data_Y = torch.Tensor(data_Y)

train_index = [i for i in range(1, 19)]
val_index = [i for i in range(19, 24)]

#train
def train(model):
    model.train()
    total_loss = 0
    random.shuffle(train_index)
    for quarter_index in train_index:
        x = data_X[quarter_index - 1].to(device)
        y = data_Y[quarter_index - 1].to(device)
        output = model(x)

        # loss = pairwise_ranking_loss(output, y)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_index) / NUMBER_OF_COMPANY
    # print("train loss:{:.6f}".format(avg_loss), end = '    ')

def val(model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for quarter_index in val_index:
            x = data_X[quarter_index - 1].to(device)
            y = data_Y[quarter_index - 1].to(device)
            output = model(x)

            # total_loss += pairwise_ranking_loss(output, y).item()
            total_loss += criterion(output, y)

    scheduler.step(total_loss)
    avg_loss = total_loss / len(val_index) / NUMBER_OF_COMPANY
    # print("val loss:{:.6f}".format(avg_loss))

#prepare model
sector_stock = dict()
for stock_num in range(NUMBER_OF_COMPANY):
    sector_num = sector_list[stock_num]
    if sector_num not in sector_stock:
        sector_stock[sector_num] = [stock_num]
    else:
        sector_stock[sector_num].append(stock_num)

intra_edge_index_src = []
intra_edge_index_tag = []
for _, value in sector_stock.items():
    for src_node in value:
        for tag_node in value:
            intra_edge_index_src.append(src_node)
            intra_edge_index_tag.append(tag_node)

inter_edge_index_src = []
inter_edge_index_tag = []
for src_node in range(len({s for s in sector_list})):
    for tag_node in range(len({s for s in sector_list})):
        inter_edge_index_src.append(src_node)
        inter_edge_index_tag.append(tag_node)

if ifDummy:
    model = GAT(NumberOfFeatures + 8, 20, 1, torch.tensor([intra_edge_index_src, intra_edge_index_tag]), torch.tensor([inter_edge_index_src, inter_edge_index_tag]), cat=GAT_cat, dropout=0.2)
else:
    model = GAT(NumberOfFeatures, 20, 1, torch.tensor([intra_edge_index_src, intra_edge_index_tag]), torch.tensor([inter_edge_index_src, inter_edge_index_tag]), cat=GAT_cat, dropout=0.2)
model = model.to(device)
lr = 0.001
epoch = 50
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

# pbar = tqdm(range(epoch))
for _ in range(epoch):
    train(model)
    val(model)
    # pbar.update()

root = config['GAT']['model_path_root']
if not os.path.isdir(root):
    os.makedirs(root)
torch.save(model.state_dict(), root + '/' + sys.argv[1] + '.pth')