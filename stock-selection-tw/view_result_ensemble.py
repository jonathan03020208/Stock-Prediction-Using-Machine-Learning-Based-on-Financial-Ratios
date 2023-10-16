## calculate return from 2019-3~2022-1
## python view_result_ensemble.py 20 4

import pickle
import sys
import numpy as np
import pandas as pd

NUMBER_OF_COMPANY = 97
number_of_quarters_per_stock = 37

input = pd.read_csv('ratio-2022-12-13.csv', dtype='str', keep_default_na=False)

return_raw = input['Stock Return'].astype(float).values.tolist()

stock_return = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    stock_return.append(return_raw[i : i + number_of_quarters_per_stock])

tw_return = return_raw[-number_of_quarters_per_stock:]

with open("company_buy_index_1.pickle", "rb") as fp:
    company_buy_RF = pickle.load(fp)
with open("company_buy_index_2.pickle", "rb") as fp:
    company_buy_FNN = pickle.load(fp)
with open("company_buy_index_3.pickle", "rb") as fp:
    company_buy_GRU = pickle.load(fp)
with open("company_buy_index_4.pickle", "rb") as fp:
    company_buy_GAT = pickle.load(fp)

ensemble_list = [company_buy_RF, company_buy_FNN, company_buy_GRU, company_buy_GAT]
# ensemble_list = [company_buy_RF, company_buy_FNN, company_buy_GRU]

#calculate return from company_buy
def cal_return(ensemble_list, topk, evK):
    num_quarters = len(company_buy_RF)

    count = []
    for i in range(num_quarters):
        quarterly_count = {}
        for company_buy in ensemble_list:
            for j in range(topK):
                if company_buy[i][j] not in quarterly_count:
                    quarterly_count[company_buy[i][j]] = 1
                else:
                    quarterly_count[company_buy[i][j]] += 1
        count.append(quarterly_count)

    # print(count)

    portfolio_return = 1.0
    for i in range(num_quarters):
        quarterly_return = 0.0
        pick_number = 0

        for index in count[i]:
            if count[i][index] >= evK:
                quarterly_return += stock_return[index][26 + i]
                pick_number += 1
        
        print(pick_number)

        if pick_number > 0:
            quarterly_return /= pick_number
            portfolio_return *= (1 + quarterly_return)
    return round(portfolio_return, 6)

def cal_return_of_0050():
    return_0050 = 1.0 
    for i in range(11):
        return_0050 *= (1 + tw_return[26 + i])
    return return_0050

topK = int(sys.argv[1])
evK = int(sys.argv[2])

#calculate portfolio return of ranked groups
print('\n0050 accumulative return from 2019-3 to 2022-1')
print('---------------------------------------------------')
print(round(cal_return_of_0050(), 6))

print('portfolio accumulative return from 2019-3 to 2022-1')
print('---------------------------------------------------')
print(cal_return(ensemble_list, topK, evK))