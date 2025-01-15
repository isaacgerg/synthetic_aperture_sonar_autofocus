import os

import numpy as np
import matplotlib.pyplot as plt

import sas_tools


fn = r'C:\Users\laptop\Downloads\Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)\Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)\MLSP-Net Data\test.npz'
output_dir = r'C:\Users\laptop\Downloads\af'

arr = np.load(fn)
x_test = arr['x_test']

# Split images into quadrants
#reshaped_tensor = x_test.reshape((66, 2, 256, 2, 256, 1))
#transposed_tensor = reshaped_tensor.transpose((1, 3, 0, 2, 4, 5))
#x_test = transposed_tensor.reshape((-1, 256, 256, 1)) 

N = x_test.shape[0]

for shadow_flag in [True, False]:
    for k in range(N):
        iut_slc = x_test[k,:,:]
        iut_drc = sas_tools.schlick(np.abs(iut_slc))
        iut_slc_pga, phi_hat, rms = sas_tools.pga(iut_slc, win = 'custom', win_params = [10, 1], shadow_pga = shadow_flag)
        iut_drc_pga = sas_tools.schlick(np.abs(iut_slc_pga))
        
        fig = plt.figure()
        plt.plot(phi_hat)
        plt.xlabel('Azimuth Bin')
        plt.ylabel('Phase Error [rad]')
        plt.tight_layout(pad=  1.01)
        plot = sas_tools.get_fig_as_numpy(fig)
        plot = np.mean(plot.astype('float32')[:,:,0:3]/255.0, axis=-1)
        plt.close('all')
        
        fig = plt.figure()
        plt.plot(rms)
        plt.ylabel('RMS [rad]')
        plt.xlabel('Iteration')
        plt.tight_layout(pad =  1.01)
        plot_rms = sas_tools.get_fig_as_numpy(fig)
        plot_rms = np.mean(plot_rms.astype('float32')[:,:,0:3]/255.0, axis=-1)    
        plt.close('all')
        
        vert_line = np.zeros((512,1))
        sas_tools.imwrite(np.hstack([iut_drc, vert_line, iut_drc_pga, vert_line, plot, vert_line, plot_rms]), os.path.join(output_dir, 'image_{}_shadow_pga_{}.png'.format(k, shadow_flag)))

assert(False)

