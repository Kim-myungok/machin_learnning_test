import data_preprocessing as dp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

class model:
    def __init__(self):
        self.run()
        
    def run(self):
        self.train_test_split()
        # self.scale_transform()
        # self.gradient_boosting_classifier()
        # self.graph()
        self.good_model_search()
        
    def train_test_split(self):
        self.df = dp.data_preprocessing_run()
        print('isnull(): ', self.df.isnull().sum())
        # print(np.isfinite(self.df))
        df_columns = self.df.columns.to_list()
        print(df_columns[1:])
        print(df_columns[:1])
        self.data_input = self.df[df_columns[1:]].to_numpy()
        self.data_target = self.df[df_columns[:1]].to_numpy()

        self.train_input, self.test_input, self.train_target, self.test_target = train_test_split(
            self.data_input, self.data_target)

    def good_model_search(self):
        
        train_score = []
        test_score = []

        for i in range(1,100) :
            hgb = HistGradientBoostingClassifier(max_iter = i, random_state=42)
        # lgb = LGBMClassifier(random_state=42)
            scores = cross_validate(hgb, self.train_input, self.train_target, return_train_score=True, n_jobs=-1)
            print(np.mean(scores['train_score']), np.mean(scores['test_score']))
            train_score.append(np.mean(scores['train_score']))
            test_score.append(np.mean(scores['test_score']))

        plt.plot(train_score)
        plt.plot(test_score)
        plt.xlabel("max_iter")
        plt.ylabel("score")
        plt.show()

if __name__ == '__main__':
    # 데이터 수집 및 각 데이터에 0이 포함된 데이터를 제외하여 각 패턴별로 25개씩 묶은 데이터세트
    df = model()

