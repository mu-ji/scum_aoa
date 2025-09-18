import numpy as np
import matplotlib.pyplot as plt
import interpolation
import music
import grid_search
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

data = np.load('SCuM_darkroom_experiment/10_data.npz')

I_data_array = data['I_data'][:40]
Q_data_array = data['Q_data'][:40]
rssi_array = data['rssi'][:40]
music_angle_array = data['music_angle'][:40]
grid_search_angle_array = data['grid_search_angle'][:40]
angle_change_1us_array = data['angle_change_1us'][:40]

music_angle_list = []
grid_search_angle_list = []


for i in range(len(I_data_array)):
    I_data = I_data_array[i]
    Q_data = Q_data_array[i]
    rssi = rssi_array[i]

    reference_I = I_data[:8]
    reference_Q = Q_data[:8]
    reference_I, reference_Q = ant_IQ_norm(reference_I, reference_Q)
    

    ref_theta_array = np.zeros(7)
    for j in range(7):
        ref_theta_array[j] = calculate_angle(reference_I[j], reference_Q[j], reference_I[j+1], reference_Q[j+1])

    angle_change_1us = np.mean(ref_theta_array)
    
    music_angle = music.cal_angle_with_music(I_data, Q_data, angle_change_1us)

    I_data_copy = np.copy(I_data)
    Q_data_copy = np.copy(Q_data)
    grid_search_angle = grid_search.cal_angle_with_grid_search(I_data_copy, Q_data_copy, angle_change_1us)

    music_angle_list.append(music_angle)
    grid_search_angle_list.append(grid_search_angle)


fig, axs = plt.subplots(3, 1, figsize=(8, 10))

axs[0].plot([i for i in range(len(I_data_array))], rssi_array, color='blue')
axs[0].set_title('rssi')
axs[0].set_xlabel('packet id')
axs[0].set_ylabel('rssi')
axs[0].grid(True)

# 第二个子图
axs[1].plot([i for i in range(len(I_data_array))], music_angle_list, color='red')
axs[1].set_title('music angle')
axs[1].set_xlabel('packet id')
axs[1].set_ylabel('angle')
axs[1].grid(True)
axs[1].set_ylim(-100, 100)
axs[1].set_yticks(np.arange(-100, 101, 10))

# 第三个子图
axs[2].plot([i for i in range(len(I_data_array))], grid_search_angle_list, color='green')
axs[2].set_title('grid search angle')
axs[2].set_xlabel('packet id')
axs[2].set_ylabel('angle')
axs[2].set_ylim(-100, 100)
axs[2].set_yticks(np.arange(-100, 101, 10))
axs[2].grid(True)


plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 第一个子图：小提琴图
parts = axs[0].violinplot([grid_search_angle_list, music_angle_list], showmeans=True, showmedians=False)

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
axs[1].boxplot([grid_search_angle_list, music_angle_list], positions=[1, 2])

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

packet_id = 5

I_data = I_data_array[packet_id]
Q_data = Q_data_array[packet_id]
angle_change_1us = angle_change_1us_array[packet_id]

x_n = interpolation.generate_xn(I_data, Q_data)
ant0_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 0)
ant1_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 1)
ant2_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 2)

ref_theta_array = np.zeros(7)
for i in range(len(I_data_array)):
    I_data = I_data_array[i]
    Q_data = Q_data_array[i]
    rssi = rssi_array[i]

    reference_I = I_data[:8]
    reference_Q = Q_data[:8]
    reference_I, reference_Q = ant_IQ_norm(reference_I, reference_Q)
    
    for j in range(7):
        ref_theta_array[j] = calculate_angle(reference_I[j], reference_Q[j], reference_I[j+1], reference_Q[j+1])

    angle_change_1us = np.mean(ref_theta_array)


print(ant0_signal[0])

plt.figure()
plt.plot([i for i in range(len(ant0_signal))], [np.arctan2(ant0_signal[i].I_data, ant0_signal[i].Q_data) for i in range(len(ant0_signal))])
plt.plot([i for i in range(len(ant0_signal))], [np.arctan2(ant1_signal[i].I_data, ant1_signal[i].Q_data) for i in range(len(ant1_signal))])
plt.plot([i for i in range(len(ant0_signal))], [np.arctan2(ant2_signal[i].I_data, ant2_signal[i].Q_data) for i in range(len(ant2_signal))])
plt.grid()
plt.legend(['ant0', 'ant1', 'ant2'])
plt.show()

plt.figure()
plt.plot([i for i in range(7)], ref_theta_array)
plt.ylim(-np.pi, np.pi)
plt.show()

angle_list = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

sample_number = 47
music_result = np.zeros((11,sample_number))
grid_search_result = np.zeros((11,sample_number))
for index in range(len(angle_list)):
    data = np.load('SCuM_darkroom_experiment/{}_data.npz'.format(angle_list[index]))
    I_data_array = data['I_data'][:sample_number]
    Q_data_array = data['Q_data'][:sample_number]
    rssi_array = data['rssi'][:sample_number]
    music_angle_array = data['music_angle'][:sample_number]
    grid_search_angle_array = data['grid_search_angle'][:sample_number]

    music_result[index,:] = music_angle_array
    grid_search_result[index,:] = grid_search_angle_array

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].boxplot(music_result.T)
axs[0].set_xticklabels(angle_list)
axs[0].set_title('music')
axs[0].set_xlabel('true angle')
axs[0].set_ylabel('angle estimation (degree)')
axs[0].set_ylim(-100, 100)
axs[0].set_yticks(np.arange(-90, 101, 10))
axs[0].plot([i/10+6 for i in angle_list], angle_list, c='b', label='perfect')
axs[0].plot([i/10+6 for i in angle_list], [i + 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
axs[0].plot([i/10+6 for i in angle_list], [i - 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
axs[0].plot([i/10+6 for i in angle_list], [i + 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
axs[0].plot([i/10+6 for i in angle_list], [i - 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
axs[0].grid(True)

axs[1].boxplot(grid_search_result.T)
axs[1].set_xticklabels(angle_list)
axs[1].set_title('grid search')
axs[1].set_xlabel('true angle')
axs[1].set_ylabel('angle estimation (degree)')
axs[1].set_ylim(-100, 100)
axs[1].set_yticks(np.arange(-90, 101, 10))
axs[1].plot([i/10+6 for i in angle_list], angle_list, c='b', label='perfect')
axs[1].plot([i/10+6 for i in angle_list], [i + 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
axs[1].plot([i/10+6 for i in angle_list], [i - 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
axs[1].plot([i/10+6 for i in angle_list], [i + 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
axs[1].plot([i/10+6 for i in angle_list], [i - 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
axs[1].grid(True)

plt.tight_layout()
plt.show()

plt.figure()
plt.boxplot(grid_search_result.T)
plt.xticks(ticks=range(1, len(angle_list)+1), labels=angle_list)
plt.xlabel('true angle (degree)', fontsize=12)
plt.ylabel('estimated angle (degree)', fontsize=12)
plt.ylim(-100, 100)
plt.yticks(np.arange(-90, 101, 10))
plt.plot([i/10+6 for i in angle_list], angle_list, c='b', label='perfect')
plt.plot([i/10+6 for i in angle_list], [i + 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
plt.plot([i/10+6 for i in angle_list], [i - 10 for i in angle_list], c='r', linestyle = '--', label='10 degree boundary')
plt.plot([i/10+6 for i in angle_list], [i + 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
plt.plot([i/10+6 for i in angle_list], [i - 5 for i in angle_list], c='y', linestyle = '--', label='5 degree boundary')
plt.grid(True)

legend_elements = [
    Line2D([0], [0], color='b', label='ground truth'),
    Line2D([0], [0], color='r', linestyle='--', label='$\pm 10^\circ$ boundary'),
    Line2D([0], [0], color='y', linestyle='--', label='$\pm 5^\circ$ boundary'),
    Patch(edgecolor='black', facecolor='w', linewidth=1, label='angle estimation')
]

plt.legend(handles=legend_elements, fontsize=12)
plt.savefig('result.svg', format='svg')
plt.show()




