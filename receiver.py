import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii

import matplotlib.pyplot as plt
from math import pi, atan2, sqrt
from scipy.linalg import eig

ser = serial.Serial('COM11', 115200)

import cmath
import interpolation
import grid_search
import music
import time
import esprit

SPEED_OF_LIGHT  = 299792458
num_iterations = 50     # 进行的循环次数
iteration = 0

rawFrame = []

all_data = {
    'I_data': [],
    'Q_data': [],
    'rssi' : [],
    'music_angle': [],
    'grid_search_angle': [],
    'angle_change_1us': []
}

music_list = []
grid_search_list = []
esprit_list = []
def calculate_angle(I1, Q1, I2, Q2):
    dot_product = I1 * I2 + Q1 * Q2
    cos_theta = np.clip(dot_product, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    cross_product = I1 *Q2 - Q1 * I2
    if cross_product > 0:
        theta = theta
    else:
        theta = -theta
    return theta

def normalization(ant_I, ant_Q):
    # 计算幅度
    amplitude = np.sqrt(ant_I**2 + ant_Q**2)
    # 归一化 I 和 Q 分量
    norm_I = ant_I / amplitude
    norm_Q = ant_Q / amplitude
    return norm_I, norm_Q

def ant_IQ_norm(ant_I_array, ant_Q_array):
    for i in range(len(ant_I_array)):
        ant_I_array[i], ant_Q_array[i] = normalization(ant_I_array[i], ant_Q_array[i])
    return ant_I_array, ant_Q_array
    
num_samples = 92

while len(music_list) <= num_iterations:
# while True:
    byte  = ser.read(1)        
    rawFrame += byte

    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 4*num_samples+8:
            received_data = rawFrame[:4*num_samples]
            num_samples = 92

            I_data = np.zeros(num_samples, dtype=np.int16)
            Q_data = np.zeros(num_samples, dtype=np.int16)
            for i in range(num_samples):
                (I) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                (Q) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))
                #print(phase)
                #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                I_data[i] = I[0]
                Q_data[i] = Q[0]

            I_data = I_data.astype(np.float32)
            Q_data = Q_data.astype(np.float32)

            storage_I_data = I_data
            storage_Q_data = Q_data

            all_data['I_data'].append(I_data)
            all_data['Q_data'].append(Q_data)

            reference_I = I_data[:8]
            reference_Q = Q_data[:8]
            reference_I, reference_Q = ant_IQ_norm(reference_I, reference_Q)
            

            ref_theta_array = np.zeros(7)
            for i in range(7):
                ref_theta_array[i] = calculate_angle(reference_I[i], reference_Q[i], reference_I[i+1], reference_Q[i+1])

            #print(ref_theta_array)
            angle_change_1us = np.mean(ref_theta_array)
            #print('angle_change_1us:',angle_change_1us)
            all_data['angle_change_1us'].append(angle_change_1us)

            storage_1us = angle_change_1us

            music_angle = music.cal_angle_with_music(I_data, Q_data, angle_change_1us)

            #print('total time cost:', time.time() - start_time, 's')
            print('music_angle:', music_angle)

            music_list.append(music_angle)


            I_data_copy = np.copy(I_data)
            Q_data_copy = np.copy(Q_data)
            grid_search_angle = grid_search.cal_angle_with_grid_search(I_data_copy, Q_data_copy, angle_change_1us)

            print('grid_search_angle_1:', grid_search_angle)
            grid_search_list.append(grid_search_angle)


            # #绘制 MUSIC 谱
            # plt.figure()
            # plt.plot(np.linspace(-90, 90, 180), 10 * np.log10(music_spectrum))
            # plt.title('MUSIC Spectrum')
            # plt.xlabel('Angle (degrees)')
            # plt.ylabel('MUSIC Spectrum (dB)')
            # plt.grid()
            # plt.xlim(-90, 90)
            # plt.show()

            #esprit_angle = esprit.cal_angle_with_esprit(I_data, Q_data, angle_change_1us)
            #esprit_list.append(esprit_angle)
            #print('esprit_angle:', esprit_angle)
            #angle = grid_search.DoA_algorithm(x_n)
            #print("grid_search:", angle)
            rssi = bytes(rawFrame[-8:-4])
            rssi = int(rssi.decode('utf-8'))
            #print(iteration)
            all_data['rssi'].append(rssi)
            all_data['music_angle'].append(music_angle)
            all_data['grid_search_angle'].append(grid_search_angle)


            print(iteration, rssi, len(all_data['rssi']))
            #print('packet_number:',rawFrame[-4])
            #print('-------------------------------')

            
        rawFrame = []
        iteration = iteration + 1

    if len(all_data['I_data']) == num_iterations:
        all_data['I_data'] = np.array(all_data['I_data'])
        all_data['Q_data'] = np.array(all_data['Q_data'])
        all_data['rssi'] = np.array(all_data['rssi'])
        all_data['music_angle'] = np.array(all_data['music_angle'])
        all_data['grid_search_angle'] = np.array(all_data['grid_search_angle'])
        all_data['angle_change_1us'] = np.array(all_data['angle_change_1us'])

        np.savez('SCuM_ondesk_experiment/-40_data.npz', **all_data)
        break


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 第一个子图：小提琴图
parts = axs[0].violinplot([grid_search_list, music_list], showmeans=True, showmedians=False)

# 设置小提琴图的标签和标题
axs[0].set_title('Violin Plot')
axs[0].set_xticks([1, 2])
axs[0].set_xticklabels(['grid search', 'music'])
axs[0].set_ylabel('angle estimation (degree)')
axs[0].set_ylim(-100, 100)
axs[0].set_yticks(np.arange(-100, 101, 10))
plt.grid
# 设置小提琴颜色
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# 第二个子图：箱线图
axs[1].boxplot([grid_search_list, music_list], positions=[1, 2])

# 设置箱线图的标签和标题
axs[1].set_title('Box Plot')
axs[1].set_xticks([1, 2])
axs[1].set_xticklabels(['grid search', 'music'])
axs[1].set_ylabel('angle estimation (degree)')
axs[1].set_ylim(-100, 100)
axs[1].set_yticks(np.arange(-100, 101, 10))

# 显示图形
plt.tight_layout()
plt.grid()
plt.show()