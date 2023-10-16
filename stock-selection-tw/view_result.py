## calculate return from 2019-3~2022-1
## python view_result.py 20

import pickle
import sys
import numpy as np
import pandas as pd
import statistics

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
return_raw = input['Stock Return'].astype(float).values.tolist()

stock_return = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    stock_return.append(return_raw[i : i + number_of_quarters_per_stock])

tw_return = return_raw[-number_of_quarters_per_stock:]

with open("company_buy_index.pickle", "rb") as fp:
    company_buy = pickle.load(fp)
pick_interval = int(sys.argv[1])

#calculate return from company_buy
def cal_return(company_buy, left, right):
    if left > NUMBER_OF_COMPANY:
        left = NUMBER_OF_COMPANY
    if right > NUMBER_OF_COMPANY:
        right = NUMBER_OF_COMPANY
    num_quarters = len(company_buy)
    pick_number = right - left + 1
    portfolio_return = 1.0 
    for i in range(num_quarters):
        quarterly_return = 0.0
        for j in range(left - 1, right):
            quarterly_return += stock_return[company_buy[i][j]][26 + i]
        quarterly_return /= pick_number       
        portfolio_return *= (1 + quarterly_return)
    return round(portfolio_return, 6)

def cal_portfolio_score(company_buy, left, right):
    if left > NUMBER_OF_COMPANY:
        left = NUMBER_OF_COMPANY
    if right > NUMBER_OF_COMPANY:
        right = NUMBER_OF_COMPANY

    pick_number = right - left + 1
    num_quarters = len(company_buy)
    quarterly_return_list = []
    for i in range(num_quarters):
        quarterly_return = 0.0
        for j in range(left - 1, right):
            quarterly_return += stock_return[company_buy[i][j]][26 + i]
        quarterly_return /= pick_number
        quarterly_return_list.append(quarterly_return)
    mean = statistics.mean(quarterly_return_list)
    stdev = statistics.stdev(quarterly_return_list)
    portfolio_score = mean / stdev
    return round(portfolio_score, 6)

def cal_precision(company_buy, topk):
    num_quarters = len(company_buy)
    stock_return_trans = np.array(stock_return).transpose()
    num_correct = 0
    for i in range(num_quarters):
        gt_return = stock_return_trans[26 + i]
        gt_company_buy = sorted(range(len(gt_return)), key=lambda k: gt_return[k])
        gt_company_buy.reverse()
        num_correct += len(set(gt_company_buy[:topk]).intersection(company_buy[i][:topk]))
    return round(num_correct / topk / num_quarters, 6)

def cal_return_of_0050():
    return_0050 = 1.0 
    for i in range(11):
        return_0050 *= (1 + tw_return[26 + i])
    return round(return_0050, 6)

def cal_0050_portfolio_score():
    return_0050_list = []
    for i in range(11):
        return_0050_list.append(tw_return[26 + i])
    mean = statistics.mean(return_0050_list)
    stdev = statistics.stdev(return_0050_list)
    portfolio_score = mean / stdev
    return round(portfolio_score, 6)


#calculate portfolio return of ranked groups
print('portfolio accumulative return from 2019-3 to 2022-1')
print('---------------------------------------------------')
for index in range(1, NUMBER_OF_COMPANY + 1, pick_interval):
    left = index
    right = index + pick_interval - 1
    if left > NUMBER_OF_COMPANY:
        left = NUMBER_OF_COMPANY
    if right > NUMBER_OF_COMPANY:
        right = NUMBER_OF_COMPANY
    print('#{}~#{} return:{}, portfolio score:{}'.format(
        left, right, cal_return(company_buy, left, right),
        cal_portfolio_score(company_buy, left, right)))

print('---------------------------------------------------')
print('pool accumulative return from 2019-3 to 2022-1')
print('return:{}, portfolio score:{}'.
        format(cal_return(company_buy, 1, 97),
               cal_portfolio_score(company_buy, 1, 97)))

print('\n0050 accumulative return from 2019-3 to 2022-1')
print('---------------------------------------------------')
print('return:{}, portfolio score:{}'.format(cal_return_of_0050(), cal_0050_portfolio_score()))

print('\ntopk precision')
print('---------------------------------------------------')
#calculate accurary
for topk in [5, 10, 20]:
    print('top{} precision:{}'.format(topk, cal_precision(company_buy, topk)))