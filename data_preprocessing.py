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
            df = pd.read_excel("data/" + file_name + ".xlsx")
            self.df_list.append(df)

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
        join_df = self.random_extraction_of_data()
        
        return join_df
    
    def zero_preprocessing_in_the_data(self):
        # 0인 데이터를 결측치 처리하여 삭제 
        df_list = []
        for df in super().get_df_list():
            for c in df.columns:
                drop_idx = df[ df[c] == 0 ].index
                df = df.drop(drop_idx)
            # msno.matrix(df)
            # plt.show()
            df.reset_index(drop=True)
            for contact in df.isnull().sum():
                # print(contact)
                if contact != 0:
                    data_info.info[df.iat[0, 0]]['is_null'] = True
                    data_info.info[df.iat[0, 0]]['number_of_zero'] = contact
                else:
                    data_info.info[df.iat[0, 0]]['is_null'] = False
                    data_info.info[df.iat[0, 0]]['number_of_zero'] = 0
            data_info.info[df.iat[0, 0]]['len'] = len(df)
            print(data_info.info[df.iat[0, 0]])
            df_list.append(df)
        super().set_df_list(df_list)

    def random_extraction_of_data(self):
        # 랜덤으로 self.n의 수만큼 데이터 추출
        df_list = []
        for df in super().get_df_list():
            df_list.append(df.sample(n=self.number_of_data))
        
        # 데이터 리스트 합치기
        join_df = pd.DataFrame()
        for df in df_list:
            join_df = pd.concat([join_df, df], ignore_index=True)
        
        return join_df

def data_preprocessing_run(number_of_data):
    file_name_list = ['청소', '수면', '요리2', '활동']
    # 청소, 수면, 요리2, 활동에 대하여 데이터를 수집함.
    data = preprocessing(file_name_list, number_of_data)
    return data.run()

if __name__ == '__main__':
    number_of_data = 15
    df = data_preprocessing_run(number_of_data)
    print(df)
    print(df.shape)