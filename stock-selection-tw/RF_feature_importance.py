import joblib
import os 
import numpy as np
import matplotlib.pyplot as plt

feature_list = ['Current Ratio',                #0
                'Long-term Debt / Capital',     #1
                'Debt/Equity Ratio',            #2
                'Gross Margin',                 #3
                'Operating Margin',             #4
                'Pre-Tax Profit Margin',        #5
                'Net Profit Margin',            #6
                'Asset Turnover',               #7
                'Inventory Turnover Ratio',     #8
                'Receiveable Turnover',         #9
                'Days Sales In Receivables',    #10
                'ROE - Return On Equity',       #11
                'Return On Tangible Equity',    #12
                'ROA - Return On Assets',       #13
                'ROI - Return On Investment',   #14
                'Book Value Per Share',         #15 
                'Operating Cash Flow Per Share',#16
                'Free Cash Flow Per Share'      #17
                ] 
                
NumberOfFeatures = len(feature_list)
model_root = 'RF/paper/2019-3~2022-1/RF_relative'
importances = [0.0 for _ in range(NumberOfFeatures)]
for model in os.listdir(model_root):
    forest = joblib.load(model_root + '/' + model)
    partial_importances = forest.feature_importances_
    for i in range(NumberOfFeatures):
        importances[i] += partial_importances[i]
for i in range(NumberOfFeatures):
    importances[i] /= 11
importances = np.array(importances)
feature_list = np.array(feature_list)
# print(sum(importances))
# print(importances)
sorted_indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(NumberOfFeatures), importances[sorted_indices], align='center')
plt.xticks(range(NumberOfFeatures), feature_list[sorted_indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance')