import data_extraction as de

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

class data_info():
    info = {'청소': {}, '수면': {}, '요리2': {}, '활동': {}}

class data_collection():
    # 각 패턴에 대하여 데이터 수집
    def __init__(self, file_name_list):
        self.df_list = []
        
        for file_name in file_name_list:
            # print("data/" + file_name + "_min.xlsx")
            # print()
            df = pd.read_excel("data/" + file_name + "_min.xlsx")
            if df.empty == False:
            # df = pd.read_excel("data/" + file_name + ".xlsx")
                self.df_list.append(df)
            # try:
            #     df = pd.read_excel("data/" + file_name + "_min.xlsx")
            #     self.df_list.append(df)
            # except FileNotFoundError:
            #     df = de.df_extraction(file_name)
            #     df = pd.read_excel("data/" + file_name + "_min.xlsx")
        # print(self.df_list)

    def print_df_list(self):
        print(self.df_list)

    def get_df_list(self):
        return self.df_list

    def set_df_list(self, df_list):
        self.df_list = df_list

            
class preprocessing(data_collection, data_info):

    def __init__(self, file_name_list, number_of_data):
        super().__init__(file_name_list)
        self.number_of_data = number_of_data
    
    def run(self):
        self.zero_preprocessing_in_the_data()
        self.set_join_df(self.random_extraction_of_data())
    
    def zero_preprocessing_in_the_data(self):
        # 0인 데이터를 결측치 처리하여 삭제 
        df_list = []
        # print(super().get_df_list())
        for df in super().get_df_list():
            if len(df) == 0:
                continue
            # print(df.columns[23:25])
            df_columns_except = ['PM2d5_1', 'PM10_1', 'Temp_1', 'Humi_1', 'CO2_1',
                                 'PM2d5_2', 'PM10_2', 'Temp_2', 'Humi_2', 'CO2_2',
                                 'PM2d5_3', 'PM10_3', 'Temp_3', 'Humi_3', 'CO2_3']
            for c in df.columns:
<<<<<<< HEAD
                df = df.drop_duplicates()
                drop_idx = df[ df[c] == 0 ].index
                df = df.drop(drop_idx)
            # msno.matrix(df)
            # plt.show()
            df.reset_index(drop=True)
=======
                if c in df_columns_except:
                    df = df.drop_duplicates()
                    drop_idx = df[ df[c] == 0 ].index
                    df = df.drop(drop_idx)
                    # msno.matrix(df)
                    # plt.show()
                df.reset_index(drop=True)
>>>>>>> origin/code
            for contact in df.isnull().sum():
                # print(contact)
                if contact != 0:
                    data_info.info[df.iat[[0, 1]]]['is_null'] = True
                    data_info.info[df.iat[0, 1]]['number_of_zero'] = contact
                else:
                    data_info.info[df.iat[0, 1]]['is_null'] = False
                    data_info.info[df.iat[0, 1]]['number_of_zero'] = 0
            data_info.info[df.iat[0, 1]]['len'] = len(df)
            # print(df.iat[0, 1], data_info.info[df.iat[0, 1]])
            df_list.append(df)
        super().set_df_list(df_list)

    def random_extraction_of_data(self):
        # 랜덤으로 self.n의 수만큼 데이터 추출
        df_list = []
        for df in super().get_df_list():
            df_list.append(df.sample(n=self.number_of_data, random_state=42))
        
        # 데이터 리스트 합치기
        join_df = pd.DataFrame()
        for df in df_list:
            join_df = pd.concat([join_df, df], ignore_index=True)
        
        return join_df

    def get_join_df(self):
        return self.join_df
    
    def set_join_df(self, df):
        self.join_df = df

class run(preprocessing):

    def __init__(self, number_of_data):
        file_name_list = ['청소', '수면', '요리2', '활동']
        super().__init__(file_name_list, number_of_data)
        self.data_preprocessing_run()

    def data_preprocessing_run(self):
        # 청소, 수면, 요리2, 활동에 대하여 데이터를 수집함.
        super().run()

if __name__ == '__main__':
    number_of_data = 250
    df = run(number_of_data).get_join_df()
    print(df)
    print(df.shape)