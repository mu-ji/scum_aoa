import numpy as np
import matplotlib.pyplot as plt
import interpolation
import cmath

SPEED_OF_LIGHT  = 299792458

def esprit(x_n):
    x_n_complex = np.zeros(x_n.shape, dtype=complex)
    for i in range(x_n.shape[0]):
       for j in range(x_n.shape[1]):
           x_n_complex[i, j] = cmath.exp(1j * cmath.phase(complex(x_n[i, j].I_data, x_n[i, j].Q_data)))
    
    z_0 = x_n_complex[:2, :]
    z_1 = x_n_complex[1:3, :]

    Z = np.row_stack((z_0, z_1))
    #print(Z.shape)
    Z_rank = np.linalg.matrix_rank(Z)
    #print(Z_rank)

    U, S, VT = np.linalg.svd(Z)
    U = U[:,:Z_rank]
    Ux = U[:-1,:]
    Uy = U[1:,:]
    Ux_pinv = np.linalg.pinv(Ux)
    Ux_pinvUy = Ux_pinv@Uy
    eigenvalues, eigenvectors = np.linalg.eig(Ux_pinvUy)

    #print(eigenvalues)

    #sin_theta = np.log(eigenvalues)*SPEED_OF_LIGHT/(-j*2*np.pi*2.40225e9*0.0375)
    #angle = np.arcsin(sin_theta)
    atan = np.arctan(eigenvalues[0].imag/eigenvalues[0].real)
    sin_theta = atan*SPEED_OF_LIGHT/(2*np.pi*2.40225e9*0.0375)
    angle = np.arcsin(sin_theta)
    angle = angle/(2*np.pi)*360
    #print(angle)
    return angle

def cal_angle_with_esprit(I_data, Q_data, angle_change_1us):

    x_n = interpolation.generate_xn(I_data, Q_data)
    ant0_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 0)
    ant1_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 1)
    ant2_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 2)
    ant3_signal = interpolation.recover_one_ant(x_n, angle_change_1us, 3)

    x_n[0,:] = ant0_signal
    x_n[1,:] = ant1_signal
    x_n[2,:] = ant2_signal
    x_n[3,:] = ant3_signal

    angle = esprit(x_n)
    return angle