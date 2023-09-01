import numpy as np
import os

loss_path = '../experimental_comparison/lr_0.0001_adjust_step_400_epoch_2000/loss_figure.csv'
loss_detail = np.loadtxt(loss_path, delimiter=',', dtype=np.float32)
loss_epoch = np.mean(loss_detail, axis=1)

np.savetxt(os.path.join('../experimental_comparison/lr_0.0001_adjust_step_400_epoch_2000', 'loss_epoch.csv'),
           loss_epoch,
           fmt='%.32f',
           delimiter=',')









