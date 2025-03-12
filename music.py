import numpy as np
import matplotlib.pyplot as plt
import cmath
import interpolation

SPEED_OF_LIGHT  = 299792458
def music_algorithm(signal_array, num_sources, angle_range):

    x_n_complex = np.zeros(signal_array.shape, dtype=complex)
    for i in range(signal_array.shape[0]):
       for j in range(signal_array.shape[1]):
           x_n_complex[i, j] = cmath.exp(1j * cmath.phase(complex(signal_array[i, j].I_data, signal_array[i, j].Q_data)))

    # 计算协方差矩阵
    cov_matrix = np.cov(x_n_complex)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 将特征值按降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 噪声子空间
    noise_space = eigenvectors[:, num_sources:]
    
    # 计算 MUSIC 谱
    music_spectrum = np.zeros(len(angle_range))
    
    for i, angle in enumerate(angle_range):
        steering_vector = np.exp(1j * (2 * np.pi * 2.40225e9 / SPEED_OF_LIGHT) * 0.0375 * np.arange(signal_array.shape[0]) * np.sin(np.radians(angle)))
        music_spectrum[i] = 1 / np.abs(steering_vector.conj().T @ noise_space @ noise_space.conj().T @ steering_vector)
    
    return music_spectrum, np.argmax(music_spectrum)-90

def cal_angle_with_music(I_data, Q_data, angle_change_1us):
    
    x_n = interpolation.generate_xn(I_data, Q_data)
    ant0_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 0)
    ant1_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 1)
    ant2_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 2)
    ant3_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 3)

    x_n[0,:] = ant0_signal
    x_n[1,:] = ant1_signal
    x_n[2,:] = ant2_signal
    x_n[3,:] = ant3_signal

    music_spectrum, music_angle = music_algorithm(x_n[0:3], 1, np.linspace(-90, 90, 180))

    return music_angle