import data_preprocessing as dp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_DATA = 25

class machine_learning_model():
    def __init__(self):
        self.run()
        
    def run(self):
        self.train_test_split()
        self.scale_transform()
        self.sgd_classifier()
        # self.good_model_search()
        
    def train_test_split(self):
        # 훈련세트와 테스트세트 나누기
        self.df = dp.run(NUMBER_OF_DATA).get_join_df()
        print(self.df)

        df_columns = self.df.columns.to_list()
        # print(df_columns[1:]) # ['PM2d5_1', 'CO2_1', 'PM2d5_2', 'CO2_2', 'PM2d5_3', 'CO2_3']
        # print(df_columns[:1]) # ['Pattern']
        self.data_input = self.df[df_columns[1:]].to_numpy()
        self.data_target = self.df[df_columns[:1]].to_numpy()

        self.train_input, self.test_input, self.train_target, self.test_target = train_test_split(
            self.data_input, self.data_target)

    def scale_transform(self):
        # 특성 스케일 맞추기
        ss = StandardScaler()
        ss.fit(self.train_input)
        self.train_scaled = ss.transform(self.train_input)
        self.test_scaled = ss.transform(self.test_input)

    def sgd_classifier(self):
        sc = SGDClassifier(loss='log', max_iter=317, tol=None)
        sc.fit(self.train_scaled, self.train_target)
        print("sc.score ", sc.score(self.train_scaled, self.train_target))

        sc.partial_fit(self.train_scaled, self.train_target)
        print(sc.score(self.train_scaled, self.train_target))
        print(sc.score(self.test_scaled, self.test_target))

    def good_model_search(self):
        sc = SGDClassifier(loss='log', random_state=42)
        train_score = []
        test_score = []
        classes = np.unique(self.train_target) # 7개의 생선의 종류
        #partial_fit() 메서드만 사용하려면 훈련 세트의 전체클래스의 레이블을 전달해 주어야 함

        for _ in range(0, 500) :
            sc.partial_fit(self.train_scaled, self.train_target, classes = classes)
            train_score.append(sc.score(self.train_scaled, self.train_target))
            test_score.append(sc.score(self.test_scaled, self.test_target))

        plt.plot(train_score)
        plt.plot(test_score)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()

if __name__ == '__main__':
    # 데이터 수집 및 각 데이터에 0이 포함된 데이터를 제외하여 각 패턴별로 25개씩 묶은 데이터세트
    df = machine_learning_model()

