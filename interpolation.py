import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import zscore
from collections import namedtuple

SAMPLE_NUMBER = 80

def normalization(ant_I, ant_Q):
    # 计算幅度
    amplitude = np.sqrt(ant_I**2 + ant_Q**2)
    # 归一化 I 和 Q 分量
    norm_I = ant_I / amplitude
    norm_Q = ant_Q / amplitude
    return norm_I, norm_Q

iq_sample = namedtuple('iq_sample', ['I_data', 'Q_data'])
def generate_xn(I_data, Q_data):
    # 创建一个包含 iq_sample 的 NumPy 数组
    x_n = np.zeros((4, SAMPLE_NUMBER), dtype=object)

    for i in range(0, SAMPLE_NUMBER, 8):
        new_iq = normalization(I_data[9:92:8][int(i/8)], Q_data[9:92:8][int(i/8)])
        x_n[0, i] = iq_sample(new_iq[0], new_iq[1])
        new_iq = normalization(I_data[11:92:8][int(i/8)], Q_data[11:92:8][int(i/8)])
        x_n[1, i + 2] = iq_sample(new_iq[0], new_iq[1])
        new_iq = normalization(I_data[13:92:8][int(i/8)], Q_data[13:92:8][int(i/8)])
        x_n[2, i + 4] = iq_sample(new_iq[0], new_iq[1])
        new_iq = normalization(I_data[15:92:8][int(i/8)], Q_data[15:92:8][int(i/8)])
        x_n[3, i + 6] = iq_sample(new_iq[0], new_iq[1])

    return x_n

# def ant_IQ_norm(ant_I_array, ant_Q_array):
#     for i in range(len(ant_I_array)):
#         ant_I_array[i], ant_Q_array[i] = normalization(ant_I_array[i], ant_Q_array[i])
#     return ant_I_array, ant_Q_array
    
def rotate_vector(I, Q, theta):
    I_new = I * np.cos(theta) - Q * np.sin(theta)
    Q_new = I * np.sin(theta) + Q * np.cos(theta)
    
    return I_new, Q_new
    
def recover_one_ant(x_n, angle_change_1us, ant_id):
    ant0_array = np.zeros((int(SAMPLE_NUMBER/8), SAMPLE_NUMBER), dtype=object)
    j = 0
    for i in range(SAMPLE_NUMBER):#x_n[ant_id,:]:
        if x_n[ant_id,:][i] != 0:
            #ant0_array[j, x_n[0,:].index(i)] = i
            #index = np.where(x_n[ant_id, :] == i)[0][0]
            ant0_array[j, i] = x_n[ant_id,:][i]
            j = j + 1

    for row in range(ant0_array.shape[0]):
        non_zero_indices = np.nonzero(ant0_array[row])[0]
        if non_zero_indices.size > 0:
            first_non_zero_index = non_zero_indices[0]
            #print('first_non_zero_index:', first_non_zero_index)
            for col in range(first_non_zero_index-1, -1, -1):
                I_new, Q_new = rotate_vector(ant0_array[row, col + 1].I_data, ant0_array[row, col + 1].Q_data, -angle_change_1us)
                ant0_array[row, col] = iq_sample(I_new, Q_new)
                
            for col in range(first_non_zero_index+1,ant0_array.shape[1]):
                I_new, Q_new = rotate_vector(ant0_array[row, col - 1].I_data, ant0_array[row, col - 1].Q_data, angle_change_1us)
                ant0_array[row, col] = iq_sample(I_new, Q_new)


    def cal_mean(ant0_array):
        mean_ant0_signal = np.zeros(SAMPLE_NUMBER, dtype=object)

        for j in range(ant0_array.shape[1]):
            I_mean = np.mean([ant0_array[i, j].I_data for i in range(ant0_array.shape[0])])
            Q_mean = np.mean([ant0_array[i, j].Q_data for i in range(ant0_array.shape[0])])
            mean_ant0_signal[j] = iq_sample(I_mean, Q_mean)
        return mean_ant0_signal

    mean_ant0_signal = cal_mean(ant0_array)

    # plt.figure()
    # colors = cm.viridis(np.linspace(0, 1, ant0_array.shape[0]))
    # for i in range(ant0_array.shape[0]):
    #     plt.plot([i for i in range(len(mean_ant0_signal))], [cmath.phase(complex(ant0_array[i,j].I_data, ant0_array[i,j].Q_data)) for j in range(len(ant0_array[i,:]))], marker='.', markersize=5, color = colors[i])

    # plt.plot([i for i in range(len(mean_ant0_signal))], [cmath.phase(complex(mean_ant0_signal[i].I_data, mean_ant0_signal[i].Q_data)) for i in range(len(mean_ant0_signal))], marker='.', markersize=5, c = 'r')

    # #plt.scatter([i for i in range(len(x_n[ant_id, :]))], x_n[ant_id, :], marker='*')
    # plt.show()

    return mean_ant0_signal

    
