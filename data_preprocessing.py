import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

class data_collection:
    # private 변수 선언
    __df = pd.DataFrame()
    __df25 = pd.DataFrame()
    n = 100

    def __init__(self, file_name):
        self.__df = pd.read_excel( 'data/' + file_name + '.xlsx' )
        self.run()

    def run(self):
        self.zero_preprocessing_in_the_data()
        self.random_extraction_of_25_data()

    def zero_preprocessing_in_the_data(self):
        # print(self.)
        for i in range(0, len(self.__df)):
            for j in range(1, len(self.__df.columns)):
                if self.__df.iat[i, j] == 0:
                    self.__df.iat[i, j] = 'NaN'
                    break

        print(self.__df.isnull().sum())
        msno.matrix(self.__df)
        plt.show()
        self.__df.dropna()
        print(self.__df)
        print(self.__df.isnull().sum())
        pass
    
    def random_extraction_of_25_data(self):
        self.set_df25(self.__df.sample(n=self.n, random_state=42))
        # print(self.__df25)
    
    def print_df(self):
        print(self.__df)

    def get_df(self):
        return self.__df

    def print_df25(self):
        print(self.__df25)

    def get_df25(self):
        return self.__df25

    def set_df25(self, df):
        self.__df25 = df

class data_join:
    __df_join = pd.DataFrame()

    def __init__(self, df_list):
        self.join(df_list)

    def join(self, class_data_list):
        for class_data in class_data_list:
            self.set_df_join(pd.concat([self.__df_join, class_data.get_df25()], ignore_index=True))
        # print(self.__df_join)

    def get_df_join(self):
        return self.__df_join

    def set_df_join(self, df):
        self.__df_join = df

def data_preprocessing_run():
    pattern = ['청소', '수면', '요리2', '활동']
    # 청소, 수면, 요리2, 활동에 대하여 데이터를 수집함.
    data_collection_class_list = []
    for p in pattern:
        data_collection_class_list.append(data_collection(p))
    # for i in range(len(data_collection_list)):
    #     data_collection_list[i].print_df25()

    data_join_collection_class = data_join(data_collection_class_list)
    df_join = data_join_collection_class.get_df_join()

    # 객체해제
    for data_collection_class in data_collection_class_list:
        del data_collection_class
    del data_join_collection_class

    
    return df_join

if __name__ == '__main__':
    print(data_preprocessing_run())