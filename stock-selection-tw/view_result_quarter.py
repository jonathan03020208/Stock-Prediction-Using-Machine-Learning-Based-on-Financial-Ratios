## calculate return from 2019-3~2022-1
## python view_result_quarter.py 20

import pickle
import sys
import numpy as np
import pandas as pd
import statistics

NUMBER_OF_COMPANY = 97
number_of_quarters_per_stock = 37

input = pd.read_csv('ratio-2022-12-13.csv', dtype='str', keep_default_na=False)

return_raw = input['Stock Return'].astype(float).values.tolist()

stock_return = []
for i in range(0, number_of_quarters_per_stock * NUMBER_OF_COMPANY, number_of_quarters_per_stock):
    stock_return.append(return_raw[i : i + number_of_quarters_per_stock])

tw_return = return_raw[-number_of_quarters_per_stock:]

with open("company_buy_index.pickle", "rb") as fp:
    company_buy = pickle.load(fp)
pick_interval = int(sys.argv[1])

#calculate return from company_buy
def show_return(company_buy, left, right):
    if left > NUMBER_OF_COMPANY:
        left = NUMBER_OF_COMPANY
    if right > NUMBER_OF_COMPANY:
        right = NUMBER_OF_COMPANY
    num_quarters = len(company_buy)
    pick_number = right - left + 1
    for i in range(num_quarters):
        quarterly_return = 0.0
        for j in range(left - 1, right):
            quarterly_return += stock_return[company_buy[i][j]][26 + i]
        quarterly_return /= pick_number       
        print(round(quarterly_return, 4), end=" ")
    print()

def show_0050_return():
    return_0050_list = []
    for i in range(11):
        return_0050_list.append(tw_return[26 + i])
    for i in range(len(return_0050_list)):
        print(round(return_0050_list[i], 4), end=" ")
    print()

print("topK")
print(show_return(company_buy, 1, pick_interval))
print("EW Pool")
print(show_return(company_buy, 1, 97))
print("TW50")
print(show_0050_return())