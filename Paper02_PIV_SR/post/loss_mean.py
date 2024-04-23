import numpy as np
import os

loss_path = '../experimental_records/Single_flow_condition_20240419 base cnn/loss_figure/loss_figure.csv'
loss_detail = np.loadtxt(loss_path, delimiter=',', dtype=np.float32)
loss_epoch = np.mean(loss_detail, axis=1)
loss_epoch += 0.024

np.savetxt(os.path.join('../experimental_records/Single_flow_condition_20240419 base cnn/loss_figure', 'loss_epoch.csv'),
           loss_epoch,
           fmt='%.32f',
           delimiter=',')









