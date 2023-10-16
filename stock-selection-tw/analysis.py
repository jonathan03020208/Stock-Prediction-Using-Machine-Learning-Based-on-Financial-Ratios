import pickle
import sys
import numpy as np
import pandas as pd
import statistics

num_quarters = 11
num_sector = 9
NUMBER_OF_COMPANY = 97

number_of_quarters_per_stock = 37

input = pd.read_csv('ratio-2022-12-13.csv', dtype='str', keep_default_na=False)
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
sector_name = {0:'Technology',
               1:'Energy',
               2:'Consumer Cyclical',
               3:'Industrials',
               4:'Consumer Defensive',
               5:'Real Estate',
               6:'Financial Services',
               7:'Communication Services',
               8:'Basic Materials',
                }

sector_list = []
sector_num = [0 for _ in range(num_sector)]
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    sector_list.append(sector_dict[input['Sector'][i]])

return_raw = input['Stock Return'].astype(float).values.tolist()

stock_return = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    stock_return.append(return_raw[i : i + number_of_quarters_per_stock])

with open("company_buy_index_4.pickle", "rb") as fp:
    company_buy = pickle.load(fp)
model_name = 'FinGAT'
topk = 20

#Industrials: 航運三雄: 2603(長榮海運)、2609(陽明海運)、2615(萬海航運)
count = [0 for _ in range(num_sector)]
_return = [0.0 for _ in range(num_sector)]
for i in range(num_quarters):
    for j in range(NUMBER_OF_COMPANY):
        buy_index = company_buy[i][j]
        # if input['CO_ID'][buy_index * number_of_quarters_per_stock] != '2603.TW' \
        #     and input['CO_ID'][buy_index * number_of_quarters_per_stock] != '2609.TW' \
        #     and input['CO_ID'][buy_index * number_of_quarters_per_stock] != '2615.TW':
        quarterly_return = stock_return[buy_index][26 + i]
        _return[sector_list[buy_index]] += quarterly_return
        count[sector_list[buy_index]] += 1
avg_return = []
for i in range(len(count)):
    avg_return.append(_return[i] / count[i])
sorted_index = sorted(range(len(avg_return)), key=lambda k: avg_return[k])
sorted_index.reverse()
print('sector:return of pool')
for i in range(num_sector):
    print("{}:{}".format(sector_name[sorted_index[i]], round(_return[sorted_index[i]] / count[sorted_index[i]], 3)))


sector_count = [0 for _ in range(num_sector)]
for x in company_buy:
    for y in x[:topk]:
        sector_count[sector_list[y]] += 1


sorted_index = sorted(range(len(sector_count)), key=lambda k: sector_count[k])
sorted_index.reverse()
print('\n' + model_name)
for i in range(num_sector):
    print("{}:{}".format(sector_name[sorted_index[i]], round(sector_count[sorted_index[i]] / float(num_quarters), 2)))