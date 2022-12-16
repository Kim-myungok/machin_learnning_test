import data_preprocessing as dp

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from tqdm import tqdm
from tqdm import trange
from scipy.stats import mode

NUMBER_OF_DATA = 100
NUM = 100

class classifier_model:
    def __init__(self):
        pass

    def model_application(self, train_scaled, train_target, test_scaled, test_target):
        return self.sgd_classifier(train_scaled, train_target, test_scaled, test_target)

    def knn_classifier(self, train_scaled, train_target, test_scaled, test_target):
        pass

    def logistic_regression(self, train_scaled, train_target, test_scaled, test_target):
        lr = LogisticRegression(C=20, max_iter=1000)
        lr.fit(train_scaled, train_target)

        train_score = lr.score(train_scaled, train_target)
        test_score = lr.score(test_scaled, test_target)
        print(train_score)
        print(test_score)
        return train_score, test_score

    def sgd_classifier(self, train_scaled, train_target, test_scaled, test_target):
        epoch = 0
        iterec = []
        coef = []
        sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
        sc.fit(train_scaled, train_target)
        print(sc.score(train_scaled, train_target))
        print(sc.score(test_scaled, test_target))
        train_score = []
        test_score = []
        classes = np.unique(train_target)
        sleep(5)

        for _ in tqdm(range(0, NUM)) :
            sc.partial_fit(train_scaled, train_target, classes = classes)
            train_score.append(sc.score(train_scaled, train_target))
            test_score.append(sc.score(test_scaled, test_target))

        # sc.predict(self.predict)
        # plt.plot(train_score, label='train_score')
        # plt.plot(test_score, label='test_score')
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy")
        # plt.legend()
        # plt.show()
        return sc

    def sgdd_classifier(self, train_scaled, train_target, test_scaled, test_target):
        pass

class machine_learning(classifier_model):

    def __init__(self):
        super().__init__()

    def get_model(self):
        return self.model
        
    def get_predict(self, data):
        # 특성 공학
        data_poly = self.poly.transform(data)
        return self.model.predict(data_poly)

    def test(self):
        test_df = pd.read_excel('test_input.xlsx').to_numpy()
        predict = self.get_predict(test_df)

        mode_ = str(mode(predict)[0][0])
        print("    >>학습한 모델이 테스트 샘플을 예측한 값은 '" + mode_ + "' 입니다.")
        print("\tㄴ " + str(mode(predict)[1][0]) + "(" + str(round(mode(predict)[1][0]/len(test_df)*100, 2)) + " %)")

    def model_score(self):
        print("train_score =", self.model.score(self.train_poly, self.train_target))
        print("test_score =", self.model.score(self.test_poly, self.test_target))
        
    def run(self):
        # while self.NUMBER_OF_DATA <= 10000:
        print('\t\t>> TOTAL_NUMBER_OF_DATA =', NUMBER_OF_DATA*4)
        self.train_test_split()
        self.scale_transform()
        self.model = super().model_application(self.train_poly, self.train_target, self.test_poly, self.test_target)
            # break
            # self.NUMBER_OF_DATA = self.NUMBER_OF_DATA + 25
            # self.sgd_classifier()
        # self.good_model_search()
        print('__학습 완료__')
        self.model_score()
        self.test()
        
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
        # 특성 공학
        self.poly = PolynomialFeatures(include_bias=False)
        self.poly.fit(self.train_input)

        self.train_poly = self.poly.transform(self.train_input)
        self.test_poly = self.poly.transform(self.test_input)
        # super().predict = poly.transform(np.array([[1.3, 665, 0.7, 694, 1.9, 697]]))
        

    def good_model_search(self):
        sc = SGDClassifier(loss='log', random_state=42)
        train_score = []
        test_score = []
        classes = np.unique(self.train_target)
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

    def predict(self):
        pass
if __name__ == '__main__':
    ML = machine_learning()
    ML.run()