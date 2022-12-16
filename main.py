#!/usr/bin/python3

import socket
import sys
import bitstring as bs
import hexdump as hd
import packet_revied as packet_rev
import machine_learning as ml
import pandas as pd
from scipy.stats import mode

HOST = '192.168.1.18'
PORT = 22500
TYPE = {'pd' : type(pd.DataFrame())}
class __socket__(packet_rev.rev):
    def __init__(self):
        super().__init__()
        self.ML = ml.machine_learning()
        self.ML.run()

    def run(self):
        server_socket = self.socket_connection()
        while True:
            print('\nwaiting to receive message')
            client, address = server_socket.accept()
            packet = client.recv(1024)
            
            print('received =  {} | bytes from = {}\npacket = '.format(len(packet), address), end='')
            print(packet)
            packet_list = []
            packet_temp = []
            for i, p in enumerate(packet):
                packet_temp.append(format(p, "#04x"))
                # print(format(p, "#04x"), end=' ')
                if i+2 < len(packet):
                    if format(packet[i+1], "#04x") == '0xee' and format(packet[i+2], "#04x") == '0xb5':
                        packet_list.append(packet_temp)
                        packet_temp = []
            packet_list.append(packet_temp)
            print(packet_list)

            # 패킷 분류 및 5개 이상 여부 확인
            super().packet_recived(packet_list)
            
            if len(super().get_df()) >= 5:
                print('데이터가 5개 이상입니다.')
                # 데이터 개수가 5개 이상일 경우, 머신러닝 모델에서 예측
                data = super().get_df().loc[:4].to_numpy()
                self.predict = self.ML.get_predict(data)
                print()
                print(self.predict)
                print(mode(self.predict))
                # 예측에 사용한 데이터 삭제
                # super().set_df()
            client.close()

    def socket_connection(self):
        sock = socket.socket()

        # Bind the socket to the port
        server_address = (HOST, PORT)
        print('starting up on {} port {}'.format(*server_address))
        s_name = socket.gethostname()
        print('서버 컴퓨터이름:', s_name)
        sock.bind(server_address)
        sock.listen(3)
        return sock
"""
data = ['PM2d5_1', 'CO2_1', 'PM2d5_2', 'CO2_2', 'PM2d5_3', 'CO2_3']
"""

if __name__ == '__main__':
    socket_ = __socket__()
    socket_.run()