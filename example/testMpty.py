import ptychopy
import matplotlib.pyplot as plt
import time
import numpy as np
# from matplotlib import animation
import h5py
import scipy.ndimage


# str = "./ptycho -jobID=sim512 -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=3 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0"

str= "./ptycho -jobID=IOTes256ePIE5 -algorithm=ePIE -fp=/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5 -fs=1 -hdf5path=/entry/data/data -beamSize=100e-6 \
         -qxy=276,616 -scanDims=51,51 -step=100e-9,100e-9 -i=10 -size=256 -lambda=1.408911284090909e-10 -dx_d=75e-6 -z=1.92 -dpf=51 -probeModes=2"

# str = "./ptycho jobID=IC_8.800keV_fly145_p10_s256_ts1_i50 -fp=/data2/JunjingData/fly145/data_roi0_dp_#d.h5 -fs=1 \
#      -beamSize=400.0e-9 -qxy=282.0,663.0 -scanDims=1,1800 -probeModes=10 -i=50 -rotate90=0 -sqrtData -fftShiftData \
#      -threshold=1 -size=256 -lambda=0.140891e-9 -dx_d=75e-6 -z=1.920 -dpf=1800 -hdf5path=/dp -probeGuess= -updateProbe=20 -updateModes=30 lf=/data2/JunjingData/fly145/fly145_pos.csv"

# str = "./ptycho jobID=IC_8.800keV_fly145_p10_s256_ts1_i50 -fp=/data2/JunjingData/mlhcode/fly145/fly145_data_#06d.h5 -fs=1 -beamSize=400.0e-9 \
#     -qxy=282.0,663.0 -scanDims=36,50 -probeModes=10 -i=50 -rotate90=0 -sqrtData -fftShiftData -threshold=1 -size=256 -lambda=0.140891e-9 \
#      -dx_d=75e-6 -z=1.920 -dpf=50 -hdf5path=/entry/data/data -probeGuess= -updateProbe=20 -updateModes=30 lf=/data2/JunjingData/mlhcode/fly145/fly145_pos.csv"

scanNo = 152
data_dir = '/data2/JunjingData/mlhcode/scan' + '%03d' %  (scanNo) + 'test/'
result_dir = '/data2/JunjingData/mlhcode/results/scan' + '%03d' %  (scanNo) + '/'

det_Npixel = 256
cen_x = 616; cen_y = 276

N_dp_x_input = det_Npixel
N_dp_y_input = det_Npixel
index_x_lb = (cen_x - np.floor(N_dp_x_input / 2.0)).astype(np.int)
index_x_ub = (cen_x + np.ceil(N_dp_x_input / 2.0)).astype(np.int)
index_y_lb = (cen_y - np.floor(N_dp_y_input / 2.0)).astype(np.int)
index_y_ub = (cen_y + np.ceil(N_dp_y_input / 2.0)).astype(np.int)

N_scan_y_lb = 0
N_scan_x_lb = 0

# dimension
N_scan_y = 51
N_scan_x = 51

filePath = 'entry/data/data'

resampleFactor = 1
resizeFactor = 1

fileName = data_dir + 'scan' + '%03d' % (scanNo) + '_data_' + '%06d' % (1) + '.h5'
h5_data = h5py.File(fileName, 'r')
dp_temp = h5_data[filePath][()];

# dp = np.zeros((N_scan_y * N_scan_x, dp_temp.shape[1], dp_temp.shape[2]))
dp = np.zeros((N_scan_y*N_scan_x,int(N_dp_y_input*resizeFactor),int(N_dp_x_input*resizeFactor)))

print(dp.shape)
for i in range(N_scan_y):
    fileName = data_dir + 'scan' + '%03d' % (scanNo) + '_data_' + '%06d' % (i + 1 + N_scan_y_lb) + '.h5'
    h5_data = h5py.File(fileName, 'r')
    # dp_temp = h5_data[filePath].value
    dp_temp = h5_data[filePath][()];
    # print(fileName, dp_temp.shape)

    for j in range(N_scan_x):
        index = i * N_scan_x + j
        # dp[index]=dp_temp[j][:][:]
        scipy.ndimage.interpolation.zoom(dp_temp[j+N_scan_x_lb, \
                                         index_y_lb:index_y_ub, \
                                         index_x_lb:index_x_ub],\
                                         [resizeFactor,resizeFactor],\
                                         dp[index,:,:], 1)

dp[dp < 0] = 0
dp[dp > 1e7] = 0

# ptychopy.epieinit(str)
iteration=10

# ptychopy.epieinit(jobID="ePIEIOTestr256", fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=iteration, size=256, dx_d=75e-6, z=1.92, \
#               probeModes=2)

ptychopy.epienpinit(jobID="ePIEIOTestr256", diffractionNP=dp, \
             fs=1, beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
              stepy=100e-9, lambd=1.408911284090909e-10, iter=iteration, size=256, dx_d=75e-6, z=1.92, \
              probeModes=2)

for i in range(iteration):
    ptychopy.epiestep()

# resprobe=ptychopy.epieresprobe()
# plt.matshow(abs(resprobe))
# plt.show()

resobj=ptychopy.epieresobj()
plt.matshow(abs(resobj))
plt.show()

ptychopy.epiepost()

# stra = "./ptycho -jobID=sim512 -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=3 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0"
#
# print(stra)
#
# ptychopy.epieinit(stra)
# for i in range(3):
#     ptychopy.epiestep()
#
# res=ptychopy.epieres()
# plt.matshow(abs(res))
# plt.show()
#
# ptychopy.epiepost()