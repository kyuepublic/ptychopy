# import ptychopy
# import matplotlib.pyplot as plt
import time
import numpy as np
# from matplotlib import animation
import h5py


scanNo = 152
data_dir = '/data2/JunjingData/mlhcode/scan' + '%03d' %  (scanNo) + 'test/'
result_dir = '/data2/JunjingData/mlhcode/results/scan' + '%03d' %  (scanNo) + '/'

# N_dp_x_input = det_Npixel
# N_dp_y_input = det_Npixel
# index_x_lb = (cen_x - np.floor(N_dp_x_input / 2.0)).astype(np.int)
# index_x_ub = (cen_x + np.ceil(N_dp_x_input / 2.0)).astype(np.int)
# index_y_lb = (cen_y - np.floor(N_dp_y_input / 2.0)).astype(np.int)
# index_y_ub = (cen_y + np.ceil(N_dp_y_input / 2.0)).astype(np.int)

N_scan_y_lb = 0
N_scan_x_lb = 0

# N_scan_y = 20
# N_scan_x = 20

# dimension
N_scan_y = 51
N_scan_x = 51

filePath = 'entry/data/data'
# roi = '_example10251'
# N_scan_y = 102
# N_scan_x = 51


resampleFactor = 1
resizeFactor = 1

fileName = data_dir + 'scan' + '%03d' % (scanNo) + '_data_' + '%06d' % (1) + '.h5'
h5_data = h5py.File(fileName, 'r')
dp_temp = h5_data[filePath][()];

dp = np.zeros((N_scan_y * N_scan_x, dp_temp.shape[1], dp_temp.shape[2]))
print(dp.shape)
for i in range(N_scan_y):
    fileName = data_dir + 'scan' + '%03d' % (scanNo) + '_data_' + '%06d' % (i + 1 + N_scan_y_lb) + '.h5'
    h5_data = h5py.File(fileName, 'r')

    # dp_temp = h5_data[filePath].value
    dp_temp = h5_data[filePath][()];
    print(fileName, dp_temp.shape)

    for j in range(N_scan_x):
        index = i * N_scan_x + j
        dp[index]=dp_temp[j][:][:]

dp[dp < 0] = 0
dp[dp > 1e7] = 0




# l1 = np.array([1.1,2.1,3.1])
# l2 = np.array([[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0,8.0,9.0], [3.0, 5.0, 0.0]])
# # l3 = np.array([[2,7, 1], [6, 3, 9], [1, 10, 13], [4, 2, 6]])  # Line A
# # l3 = np.array([])                                             # Line B
# l3 = np.array([[[2,7, 1, 11], [6, 3, 9, 12]],
#                  [[1, 10, 13, 15], [4, 2, 6, 2]]])
#
#
# # Whole mode key work, with numpy array as input       positionNP=l2,   objectNP=l2, probeNP=l2,
# ptychopy.epienp(jobID="ePIEIOTestr256", diffractionNP=l3, fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92, dpf=51, \
#               probeModes=1)