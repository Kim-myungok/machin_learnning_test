import pandas as pd
from datetime import datetime, timedelta

def df_extraction(file_name):
    df = pd.read_excel('data/nov_' + file_name + '.xlsx')
    # print(df)
    df_columns = df.columns
    print(df_columns)
    #print(df.to_dict())

    date_list = df["Date"].to_list()
    sec = int(date_list[0].second)
    index = [0, ]
    print("\tsec = ", sec)
    for i in range(2, len(date_list)):
        # print(date_list[i] - date_list[i-1])
        if int(date_list[i].minute) - int(date_list[i-1].minute) >= 1:
            print(date_list[i] - date_list[i-1])
            sec = int(date_list[i].second)
            index.append(i)

    df = df.loc[index]
    df = df.reset_index()
    print()
    print(df)
    print(df.shape)

    df.to_excel("data/" + file_name + "_min.xlsx")

def df_print(df):
    df_columns = df.columns
    df = df.reset_index()
    pattern_list = df["Pattern"].to_list()
    index = [0, ]
    

    for i in range(2, len(pattern_list)):
        # print(date_list[i] - date_list[i-1])
        if pattern_list[i] != pattern_list[i-1]:
            print(pattern_list[i-1], ' -> ', pattern_list[i])
            index.append(i)

    print()
    df = df.loc[index]
    df = df.reset_index()
    print(df)
    print(df.shape)
    context_save(df, 'df_print')

def context_save(df, text):
    if text == 'df_extraction':
        df.to_excel("data_extravt.xlsx")
    else:
        df.to_excel("data_print.xlsx")


if __name__ == "__main__":
    df = df_extraction()
    context_save(df, 'df_extraction')
    # df_print(df)
    '''
    < 메모 >
    1. 시계열로 1분마다 데이터 하나씩 추출
    2. 단 시계열 초단위 차이가 3초 이하이면서 모드가 같아야함.

    '''