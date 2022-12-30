import data_preprocessing as dp

import pandas as pd
# 교차검증
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# 트리 모델
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from lightgbm import LGBMClassifier

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
TARGET_SCORE = 60
TEST_INPUT_FILE_LOCAL = 'data/10월 .min/'

class classifier_model:
    def __init__(self, n_estimators, learning_rate):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def HistGradientBoostingClassifier(self, train_input, train_target):
        print("HistGradientBoostingClassifier()")
        hgb = HistGradientBoostingClassifier(max_iter = 1, random_state=42)
        # lgb = LGBMClassifier(random_state=42)
        hgb.fit(train_input, train_target)
        return hgb

    def GradientBoostingClassifier(self, train_input, train_target):
        print("GradientBoostingClassifier()")
        print("\t\t>> n_estimators =", self.n_estimators)
        print("\t\t>> learning_rate =", self.learning_rate)
        gb = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate ,random_state=42)
        # n_estimators = 트리의 개수
        gb.fit(train_input, train_target)
        return gb

    def set_model_property(self, model_property_list):
        self.n_estimators = model_property_list[0]
        self.learning_rate = model_property_list[1]

class machine_learning(classifier_model):

    def __init__(self, n_estimators,learning_rate):
        super().__init__(n_estimators, learning_rate)

    def run(self):
        # while self.NUMBER_OF_DATA <= 10000:
        # print('\t\t>> TOTAL_NUMBER_OF_DATA =', NUMBER_OF_DATA*4)
        self.train_test_split()
        # self.select_model = {'hgb':super().HistGradientBoostingClassifier(self.train_input, self.train_target),
        #                      'gb':super().GradientBoostingClassifier(self.train_input, self.train_target)}
        self.model = super().GradientBoostingClassifier(self.train_input, self.train_target)
            # break
            # self.NUMBER_OF_DATA = self.NUMBER_OF_DATA + 25
            # self.sgd_classifier()
        # self.good_model_search()
        # print('__학습 완료__')
        train_score, test_score = self.model_score()
        if train_score * 100 > TARGET_SCORE:
            if (train_score * 100) - (test_score * 100) <= 5:
                return True
            else:
                self.error(-1)
                return False
        
    def train_test_split(self):
        # 훈련세트와 테스트세트 나누기
        self.df = dp.run(NUMBER_OF_DATA).get_join_df()
        df_columns = self.df.columns.to_list()
        # print(df_columns[4:-4]) # [공기질 데이터 모두]
        # print(df_columns[:1])   # [   'Pattern'   ]
        df_columns_ = df_columns[4:]
        # print(df_columns[1], df_columns_)
        data_input = self.df[df_columns_].to_numpy()
        data_target = self.df[df_columns[1]].to_numpy()

        self.train_input, self.test_input, self.train_target, self.test_target = train_test_split(
            data_input, data_target, test_size=0.2, random_state=42)

    def model_score(self):
        scores = cross_validate(self.model, self.train_input, self.train_target, return_train_score=True, n_jobs=-1) # 훈련세트, 교차검증
        train_score = np.mean(scores['train_score'])
        test_score = np.mean(scores['test_score'])
        print("\t>> train_score =", train_score)
        print("\t>> test_score =", test_score)
        return train_score, test_score
        
    def get_model(self):
        return self.model
        
    def get_predict(self, data_input):
        return self.model.predict(data_input)

    def error(self, num):
        if num == -1:
            # print("과대적합 혹은 과소적합 발생")
            pass

    def test(self):
        print(self.df)
        train_score = self.model.score(self.train_input, self.train_target)
        test_score = self.model.score(self.test_input, self.test_target)
        print("\t>> train_score =", train_score)
        print("\t>> test_score =", test_score)

        test_pattern = input(">> 테스트할 Pattern을 입력하시오... ex) 수면, 청소, 요리2, 활동\n>> ")
        if test_pattern == 'exit' or test_pattern == '' or test_pattern == '종료' or test_pattern == 'q':
            return -1
        test_df = pd.read_excel(TEST_INPUT_FILE_LOCAL + test_pattern + "_min.xlsx")
        #랜덤으로 10개 혹은 5개 뽑아서 예측하ㄱ는것으로 수정하기
        test_df_columns = test_df.columns.to_list()
        """======================================================================================"""
        target_count = 0
        NUMBER_OF_TIMES = 100
        TEST_INPUT_NUMBER = 100
        """======================================================================================"""

        print("\t>> 일치여부\t예측한값\t목표한값\t정답확률", end='')
        df_prdt = pd.DataFrame(columns=['Target', 'Predict', 'Answer_rate', 'Match_status', 'Total_rate'])
        for i in range(NUMBER_OF_TIMES):
            # 테스트할 데이터를 넘파이 배열로 변경하여 랜덤으로 10개씩 추출하여 예측
            test_input = test_df[test_df_columns[4:-4]].to_numpy()
            test_input = np.random.permutation(test_input)
            predict = self.get_predict(test_input[:TEST_INPUT_NUMBER])

            mode_ = str(mode(predict)[0][0])
            # 일치
            if mode_ in test_pattern:
                insert_data = {'Target' : test_pattern, 'Predict' : mode_,
                               'Answer_rate' : str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %)",
                               'Match_status': 'O'}
                target_count = target_count + 1
                if i == 99:
                    insert_data['Total_rate'] = target_count / NUMBER_OF_TIMES * 100
                df_prdt = df_prdt.append(insert_data, ignore_index=True)
            # 불일치
            else:
                insert_data = {'Target' : test_pattern, 'Predict' : mode_,
                               'Answer_rate' : str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %)",
                               'Match_status': 'X'}
                if i == 99:
                    insert_data['Total_rate'] = target_count / NUMBER_OF_TIMES * 100
                df_prdt = df_prdt.append(insert_data, ignore_index=True)
        print(df_prdt)
        print(self.model.feature_importances_)

if __name__ == '__main__':
    for n_ in range(30, 199, 10):
        # for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            ML = machine_learning(n_, 0.1)
            sign = ML.run()
            if sign:
                break
    saved_model = pickle.dumps(ML.model)
    now_datetime = datetime.today().strftime("%Y%m%d_%H%M")
    file_name = 'saved_model\gb_' + str(now_datetime) + '.pkl'
    print(file_name)
    joblib.dump(ML.model, file_name)
    while True:
        if ML.test() == -1:
            break