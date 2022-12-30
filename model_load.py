import pandas as pd

# 기타 라이브러리
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import mode
import os
#학습한 모델 저장 및 불러오기
import joblib

TARGET_SCORE = 60
TEST_INPUT_FILE_LOCAL = 'data/10월 .min/'

class machine_learning():

    def __init__(self, file_name):
        self.model = joblib.load('saved_model/' + file_name + '.pkl')

    def run(self):
        self.df_predict_list = self.df_predict__init__()
        while True:
            if self.test() == -1:
                break
            else:
                print(self.df_predict_list)
        return self.df_predict_list
        # self.df_predict_list.to_excel("test.xlsx")

    def df_predict__init__(self):
        df = pd.DataFrame(columns=['Target', 'Predict', 'Answer_rate', 'Match_status', 'Total_rate'])
        return df
        
    def get_predict(self, data_input):
        return self.model.predict(data_input)

    def df_predict_list_join(self, add_df):
        self.df_predict_list = pd.concat([self.df_predict_list, add_df], ignore_index=True)

    def test(self):
        print()
        print()
        print("==================================================================")
        test_pattern = input(">> 테스트를 진행하시겠습니까? [ y / n ]\n>> ")
        if test_pattern == 'n' or test_pattern == '' or test_pattern == '종료' or test_pattern == 'q'  or test_pattern == 'ㅂ' or test_pattern == 'ㄷ턋':
            return -1
        for file_name in ['수면', '청소', '요리2', '활동']:
            test_df = pd.read_excel(TEST_INPUT_FILE_LOCAL + file_name + "_min.xlsx")
            #랜덤으로 10개 혹은 5개 뽑아서 예측하ㄱ는것으로 수정하기
            test_df_columns = test_df.columns.to_list()
            """======================================================================================"""
            target_count = 0
            NUMBER_OF_TIMES = 100
            TEST_INPUT_NUMBER = 100
            """======================================================================================"""

            df_prdt = self.df_predict__init__()
            for i in range(NUMBER_OF_TIMES):
                # 테스트할 데이터를 넘파이 배열로 변경하여 랜덤으로 10개씩 추출하여 예측
                test_input = test_df[test_df_columns[4:]].to_numpy()
                test_input = np.random.permutation(test_input)
                predict = self.get_predict(test_input[:TEST_INPUT_NUMBER])

                mode_ = str(mode(predict)[0][0])
                # 일치
                if mode_ in file_name:
                    data_insert = {'Target' : [file_name], 'Predict' : [mode_],
                                'Answer_rate' : [str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %"],
                                'Match_status': ['O']}
                    target_count = target_count + 1
                # 불일치
                else:
                    data_insert = {'Target' : [file_name], 'Predict' : [mode_],
                                'Answer_rate' : [str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %"],
                                'Match_status': ['X']}
                if i == 99:
                    data_insert['Total_rate'] = [str(round(target_count / NUMBER_OF_TIMES * 100, 2)) + " %"]
                    print("==================================================================")
                    print("data_insert['Target']:", data_insert['Target'])
                    print("data_insert['Total_rate']:", data_insert['Total_rate'])
                    print()
                df_data_insert = pd.DataFrame(data_insert)
                df_prdt = pd.concat([df_prdt, df_data_insert], ignore_index=True)
            # print(df_prdt)
            print(self.model.feature_importances_)
            self.df_predict_list_join(df_prdt)
        
    # def test(self):
    #     print()
    #     print()
    #     print("==================================================================")
    #     test_pattern = input(">> 테스트할 Pattern을 입력하시오... ex) 수면, 청소, 요리2, 활동\n>> ")
    #     if test_pattern == 'exit' or test_pattern == '' or test_pattern == '종료' or test_pattern == 'q'  or test_pattern == 'ㅂ' or test_pattern == 'ㄷ턋':
    #         return -1
    #     test_df = pd.read_excel(TEST_INPUT_FILE_LOCAL + test_pattern + "_min.xlsx")
    #     #랜덤으로 10개 혹은 5개 뽑아서 예측하ㄱ는것으로 수정하기
    #     test_df_columns = test_df.columns.to_list()
    #     """======================================================================================"""
    #     target_count = 0
    #     NUMBER_OF_TIMES = 100
    #     TEST_INPUT_NUMBER = 100
    #     """======================================================================================"""

    #     df_prdt = self.df_predict__init__()
    #     for i in range(NUMBER_OF_TIMES):
    #         # 테스트할 데이터를 넘파이 배열로 변경하여 랜덤으로 10개씩 추출하여 예측
    #         test_input = test_df[test_df_columns[4:-4]].to_numpy()
    #         test_input = np.random.permutation(test_input)
    #         predict = self.get_predict(test_input[:TEST_INPUT_NUMBER])

    #         mode_ = str(mode(predict)[0][0])
    #         # 일치
    #         if mode_ in test_pattern:
    #             data_insert = {'Target' : [test_pattern], 'Predict' : [mode_],
    #                            'Answer_rate' : [str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %"],
    #                            'Match_status': ['O']}
    #             target_count = target_count + 1
    #         # 불일치
    #         else:
    #             data_insert = {'Target' : [test_pattern], 'Predict' : [mode_],
    #                            'Answer_rate' : [str(round(mode(predict)[1][0]/len(test_input[:TEST_INPUT_NUMBER])*100, 2)) + " %"],
    #                            'Match_status': ['X']}
    #         if i == 99:
    #             data_insert['Total_rate'] = [str(round(target_count / NUMBER_OF_TIMES * 100, 2)) + " %"]
    #         df_data_insert = pd.DataFrame(data_insert)
    #         df_prdt = pd.concat([df_prdt, df_data_insert], ignore_index=True)
    #     # print(df_prdt)
    #     print(self.model.feature_importances_)
    #     self.df_predict_list_join(df_prdt)

if __name__ == '__main__':
    df = pd.DataFrame()
    file_name = 'gb_20221230_1111'
    while True:
        df = machine_learning(file_name).run()
        input_text = input(">> 프로그램을 종료하시겠습니까? [ y(enter) / n ]\n>> ")
        if input_text == 'y' or input_text == '':
            break
    print(">> 프로그램이 종료되었습니다.")
    print(df)
    PREDICT_SAVED_DATA_LOCAL = 'saved_model/' + file_name + ' (predict)'
    now_datetime = str(datetime.today().strftime("%Y%m%d_%H%M"))
    try:
        df.to_html(PREDICT_SAVED_DATA_LOCAL + "/predict_"+ now_datetime +".html")
        df.to_excel(PREDICT_SAVED_DATA_LOCAL + "/predict_"+ now_datetime +".xlsx")
    except OSError:
        if not os.path.exists(PREDICT_SAVED_DATA_LOCAL):
            os.makedirs(PREDICT_SAVED_DATA_LOCAL)
        df.to_html(PREDICT_SAVED_DATA_LOCAL + "/predict_"+ now_datetime +".html")
        df.to_excel(PREDICT_SAVED_DATA_LOCAL + "/predict_"+ now_datetime +".xlsx")