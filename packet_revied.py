import pandas as pd
"""
수신받은 패킷 분류작업하는 프로그램

기능코드 : 0x06
기능 : IAQ 정보
"""
class rev:
    def __init__(self):
        self.df = pd.DataFrame(columns=['PM2d5_1', 'CO2_1', 'PM2d5_2', 'CO2_2', 'PM2d5_3', 'CO2_3'])

    def df__init__(self):
        self.df = pd.DataFrame(columns=['PM2d5_1', 'CO2_1', 'PM2d5_2', 'CO2_2', 'PM2d5_3', 'CO2_3'])

    def get_df(self):
        return self.df

    def set_df(self, df):
        self.df = df

    def packet_recived(self, packet_list):
        # print("function_code_classification(packet_list) run...")

        for packet in packet_list:
            print(": IAQ 정보")
            self.iaq_sensor(packet[2:-4])

        print()
        print("[result]")
        print(self.df)
        if len(self.df) >= 5:
            return self.df

    def iaq_sensor(self, packet):
        # 데이터 패킷부분만 잘라서 매개변수로 받아온것이기 때문에
        # 첫번째 인덱스부터 데이터의 내용이 됨
        data = []
        for i in range(0, len(packet), 2):
            if ((i % 2) % 2) == 0:
                data.append((int(packet[i], 0)*0x100 + int(packet[i+1], 0))/10)
            else:
                data.append(int(packet[i], 0)*0x100 + int(packet[i+1], 0))
        print(data)
        # print(len(packet), '\t', packet)
        self.pd_data_add(data)
        # data = {'PM2d5_1' : 0, 'CO2_1' : 0, 'PM2d5_2' : 0, 'CO2_2' : 0, 'PM2d5_3' : 0, 'CO2_3' : 0}
        # data_key = list(data.keys())
        # j=0
        # for i in range(0, len(packet), 2):
        #     if 'CO2' not in data_key[j]:
        #         data[data_key[j]] = (int(packet[i], 0)*0x100 + int(packet[i+1], 0))/10
        #     else:
        #         data[data_key[j]] = int(packet[i], 0)*0x100 + int(packet[i+1], 0)
        #     # print(data_key[j], '\t', data[data_key[j]], '\t', i)
        #     j = j + 1
        # # print(len(packet), '\t', packet)
        # self.pd_data_add(data)

    def pd_data_add(self, new_row_data):
        # new_row = pd.DataFrame.from_dict([data])
        self.df.loc[len(self.df)] = new_row_data

if __name__ == '__main__':
    packet = [['0xee', '0xb5',
               '0x00', '0x0e', '0x01', '0xe8',
               '0x00', '0x1a', '0x03', '0xe8',
               '0x00', '0x1e', '0x02', '0xe8',
               '0xff', '0xfc', '0xff', '0xff']]
    data = rev()
    print(data.packet_recived(packet))
    # server.data_insert_in_sql_server(data)
    # print(server.data_load_in_sql_server())

