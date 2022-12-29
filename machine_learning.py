import data_preprocessing as dp

import pandas as pd
from sklearn.model_selection import train_test_split

# 분류 모델
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
# 기타 라이브러리
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
#학습한 모델 저장
import pickle
import joblib

from datetime import datetime
from tqdm import tqdm
from tqdm import trange
from scipy.stats import mode

NUMBER_OF_DATA = 250
NUM = 1000

TEST_INPUT_FILE_LOCAL = 'data/10월 .min/'

class classifier_model:
    def __init__(self):
        self.model_epoch = 0

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
        # print(sc.score(train_scaled, train_target))
        # print(sc.score(test_scaled, test_target))
        # train_score = []
        # test_score = []
        classes = np.unique(train_target)
        # sleep(5)
        # self.model_epoch = self.model_epoch + 10
        # print('\t\t>> self.model_epoch =', self.model_epoch)

        for _ in range(0, self.model_epoch):
            sc.partial_fit(train_scaled, train_target, classes = classes)
        # train_score.append(sc.score(train_scaled, train_target))
        # test_score.append(sc.score(test_scaled, test_target))

        # sc.predict(self.predict)
        # plt.plot(train_score, label='train_score')
        # plt.plot(test_score, label='test_score')
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy")
        # plt.legend()
        # plt.show()
        return sc

    def add_model_epoch(self, epoch):
        self.model_epoch = epoch
        print('\t\t>> add_model_epoch =', self.model_epoch)
        
class tree_model:
    def __init__(self):
        self.model_epoch = 0

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
        # print(sc.score(train_scaled, train_target))
        # print(sc.score(test_scaled, test_target))
        # train_score = []
        # test_score = []
        classes = np.unique(train_target)
        # sleep(5)
        # self.model_epoch = self.model_epoch + 10
        # print('\t\t>> self.model_epoch =', self.model_epoch)

        for _ in range(0, self.model_epoch):
            sc.partial_fit(train_scaled, train_target, classes = classes)
        # train_score.append(sc.score(train_scaled, train_target))
        # test_score.append(sc.score(test_scaled, test_target))

        # sc.predict(self.predict)
        # plt.plot(train_score, label='train_score')
        # plt.plot(test_score, label='test_score')
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy")
        # plt.legend()
        # plt.show()
        return sc

    def add_model_epoch(self, epoch):
        self.model_epoch = epoch
        print('\t\t>> add_model_epoch =', self.model_epoch)
        
class machine_learning(classifier_model):

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def get_model(self):
        return self.model
        
    def get_predict(self, data):
        # 특성 공학
        data_poly = self.poly.transform(data)
        return self.model.predict(data_poly)

    def test(self):
        print(self.df)
        train_score = self.model.score(self.train_poly, self.train_target)
        test_score = self.model.score(self.test_poly, self.test_target)
        print("train_score =", train_score)
        print("test_score =", test_score)

        test_pattern = input(">> 테스트할 Pattern을 입력하시오... ex) 수면, 청소, 요리2, 활동\n>> ")
        if test_pattern == 'exit' or test_pattern == '':
            return -1
        test_df = pd.read_excel(TEST_INPUT_FILE_LOCAL + test_pattern + "_min.xlsx")
        #랜덤으로 10개 혹은 5개 뽑아서 예측하ㄱ는것으로 수정하기
        test_df_columns = test_df.columns.to_list()
        """======================================================================================"""
        target_count = 0
        NUMBER_OF_TIMES = 100
        TEST_INPUT_NUMBER = 100
        """======================================================================================"""
        for i in range(NUMBER_OF_TIMES):
            # 테스트할 데이터를 넘파이 배열로 변경하여 랜덤으로 10개씩 추출하여 예측
            test_input = test_df[test_df_columns[4:-4]].to_numpy()
            test_input = np.random.permutation(test_input)
            predict = self.get_predict(test_input[:TEST_INPUT_NUMBER])

            mode_ = str(mode(predict)[0][0])
            print("    >> 학습한 모델이 테스트 샘플을 예측한 값은 '" + mode_ + "' 입니다.")
            if mode_ in test_pattern:
                target_count = target_count + 1
            else:
                print("\tㄴ 불일치: " + test_pattern)
            print("\tㄴ " + str(mode(predict)[1][0]) + "(" + str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %)")
        print("\tㄴ 총 맞춘 개수: "+ str(target_count) + "/" + str(NUMBER_OF_TIMES))

    def model_score(self):
        train_score = self.model.score(self.train_poly, self.train_target)
        test_score = self.model.score(self.test_poly, self.test_target)
        print("train_score =", train_score)
        print("test_score =", test_score)
        if train_score * 100 > 60:
            if -1 <= (train_score * 100) - (test_score * 100) and (train_score * 100) - (test_score * 100) <= 1:
                return False
        return True
        
    def run(self):
        # while self.NUMBER_OF_DATA <= 10000:
        # print('\t\t>> TOTAL_NUMBER_OF_DATA =', NUMBER_OF_DATA*4)
        self.train_test_split()
        self.scale_transform()
        self.model = super().model_application(self.train_poly, self.train_target, self.test_poly, self.test_target)
            # break
            # self.NUMBER_OF_DATA = self.NUMBER_OF_DATA + 25
            # self.sgd_classifier()
        # self.good_model_search()
        # print('__학습 완료__')
        if self.model_score():
            self.error(-1)
            super().add_model_epoch(self.epoch)
            return False
        else:
            return True
        
        
    def train_test_split(self):
        # 훈련세트와 테스트세트 나누기
        self.df = dp.run(NUMBER_OF_DATA).get_join_df()
        # print(self.df)

        df_columns = self.df.columns.to_list()
        # print(df_columns[1:]) # ['PM2d5_1', 'CO2_1', 'PM2d5_2', 'CO2_2', 'PM2d5_3', 'CO2_3']
        # print(df_columns[:1]) # ['Pattern']
        
        df_columns_ = df_columns[4:-4]
        print(df_columns[1], df_columns_)
        self.data_input = self.df[df_columns_].to_numpy()
        self.data_target = self.df[df_columns[1]].to_numpy()

        self.train_input, self.test_input, self.train_target, self.test_target = train_test_split(
            self.data_input, self.data_target)
        
    def scale_transform(self):
        # 특성 공학
        self.poly = PolynomialFeatures(include_bias=False)
        self.poly.fit(self.train_input)

        self.train_poly = self.poly.transform(self.train_input)
        self.test_poly = self.poly.transform(self.test_input)
        # super().predict = poly.transform(np.array([[1.3, 665, 0.7, 694, 1.9, 697]]))
        
    def error(self, num):
        if num == -1:
            # print("과대적합 혹은 과소적합 발생")
            pass
    # def good_model_search(self):
    #     sc = SGDClassifier(loss='log', random_state=42)
    #     train_score = []
    #     test_score = []
    #     classes = np.unique(self.train_target)
    #     #partial_fit() 메서드만 사용하려면 훈련 세트의 전체클래스의 레이블을 전달해 주어야 함

    #     for _ in range(0, 500) :
    #         sc.partial_fit(self.train_scaled, self.train_target, classes = classes)
    #         train_score.append(sc.score(self.train_scaled, self.train_target))
    #         test_score.append(sc.score(self.test_scaled, self.test_target))

    #     plt.plot(train_score)
    #     plt.plot(test_score)
    #     plt.xlabel("epoch")
    #     plt.ylabel("accuracy")
    #     plt.show()

if __name__ == '__main__':
    epoch = 0
    while True:
        epoch = epoch + 100
        ML = machine_learning(epoch)
        sign = ML.run()
        if sign:
            break
    saved_model = pickle.dumps(ML.model)
    now_datetime = datetime.today().strftime("%Y%m%d_%H%M")
    file_name = 'saved_model\SGD_' + str(now_datetime) + '.pkl'
    print(file_name)
    joblib.dump(ML.model, file_name)
    while True:
        if ML.test() == -1:
            break