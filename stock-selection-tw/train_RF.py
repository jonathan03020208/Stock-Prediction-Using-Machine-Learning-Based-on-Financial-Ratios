### Before running, plz setup the feild [RF] in config.ini
### python train_RF.py 2019-3

import random
import pickle
from tqdm import tqdm 
import statistics
import sys
import os
import configparser
import json
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
import joblib

config = configparser.ConfigParser()    
config.read('config.ini')

    
#ifDummy
if config['RF']['dummy'].lower() == 'true':
    ifDummy = True
elif config['RF']['dummy'].lower() == 'false':
    ifDummy = False
else:
    raise RuntimeError('dummy must be true or false')

#ifRelative
if config['RF']['relative'].lower() == 'true':
    ifRelative = True
elif config['RF']['relative'].lower() == 'false':
    ifRelative = False
else:
    raise RuntimeError('relative must be true or false')

#dir root
root = config['RF']['model_path_root'] + '/' + sys.argv[1] + '/'

NUMBER_OF_COMPANY = 97

year = int(sys.argv[1].split('-')[0])
quarter = int(sys.argv[1].split('-')[1])

# 2013-1 => #1
simulation_num = (year - 2013) * 4 + quarter

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
    ratio_scope.append(ratio[i][simulation_num-27:simulation_num])

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
    stock_return.append(return_raw[i+simulation_num-27 : i + number_of_quarters_per_stock])

tw_return = return_raw[-number_of_quarters_per_stock+simulation_num-27:]

stock_relative_return = []
for i in range(NUMBER_OF_COMPANY):
    stock_relative_return.append([stock_return[i][k] - tw_return[k] for k in range(len(stock_return[i]))])

X = []
Y = []
for i in range(NUMBER_OF_COMPANY):
    for k in range(1, 27):
        X.append(ratio_scope[i][k - 1])       
        if ifRelative:
            Y.append(stock_relative_return[i][k - 1])
        else:
            Y.append(stock_return[i][k - 1])

forest = RandomForestRegressor(n_estimators = 1000, criterion = 'squared_error')
forest.fit(X, Y)

model_path_root = config['RF']['model_path_root']
if not os.path.isdir(model_path_root):
    os.makedirs(model_path_root)

model_path = model_path_root + '/' + sys.argv[1] + '.joblib'
joblib.dump(forest, model_path)