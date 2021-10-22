import ptychopy
# import matplotlib.pyplot as plt
import time
import numpy as np
# from matplotlib import animation
import h5py
import scipy.ndimage


scanNo = 2035
data_dir = '/data2/JunjingData/mlhcode/data/brain_tile_scanning/'
result_dir = '/data2/JunjingData/mlhcode/results/brain_tile_scanning/'

det_Npixel = 128
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
N_scan_y = 2064-2034
N_scan_x = 104560

filePath = 'dp'

resampleFactor = 1
resizeFactor = 1

scanPositionFile='/data2/JunjingData/mlhcode/data/brain_tile_scanning/fly2035T64_ori_pos.csv'
sparray = np.genfromtxt(scanPositionFile, delimiter=',')
print(sparray.shape);
sumshape=0;

avgFileSize=sparray.shape[0]//N_scan_y;
remainer=sparray.shape[0]%N_scan_y;
dp = np.zeros((1*avgFileSize,int(N_dp_y_input*resizeFactor),int(N_dp_x_input*resizeFactor)));
# dp = np.zeros((1*N_scan_x,int(N_dp_y_input*resizeFactor),int(N_dp_x_input*resizeFactor)))

scanPositionFileResult=result_dir+'fly2035T36_ori_cropped_first4_pos.csv';
sparrayresult=sparray[:4*avgFileSize];
np.savetxt(scanPositionFileResult, sparrayresult, delimiter=',')

print("average file size for each new hdf5 file",dp.shape)
dpstart=0;
dpend=0;
index=0

# fileindex=0;
arryindex=0;
dpfileindex=0;
dparrayindex=0;

# this is the sum of the file szie, will increase for each loop
prefilesize=0;

for fileindex in range(N_scan_y):

    fileName = data_dir + 'fly' + '%4d' % (scanNo+fileindex) + '_Ndp128_dp.h5'

    h5_data = h5py.File(fileName, 'r')
    file_temp = h5_data[filePath][()];
    print(fileName, file_temp.shape)
    startdpfileindex=(prefilesize+1)//avgFileSize;
    startdparrindex=(prefilesize+1)%avgFileSize;

    enddpfileindex=(prefilesize+file_temp.shape[0])//avgFileSize;
    enddparrindex = (prefilesize + file_temp.shape[0])%avgFileSize;

    if startdpfileindex==enddpfileindex:
        if startdparrindex==0 and enddparrindex>=1:
            dp[avgFileSize-1:,:,:]= file_temp[:1, :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo+startdpfileindex-1) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
            dp[:enddparrindex, :, :] = file_temp[1:, :, :]
        elif startdparrindex>=1 and enddparrindex>=1:
            dp[(startdparrindex - 1):(file_temp.shape[0]+startdparrindex - 1), :, :] = file_temp[:, :, :]
            #impossible enddparrindex == 0  enddparrindex is [1, n-1]
    elif enddpfileindex==(startdpfileindex+1):
        if startdparrindex == 0 and enddparrindex == 0:
            dp[avgFileSize-1:, :, :] = file_temp[:1, :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo+startdpfileindex-1) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
            dp[:, :, :] = file_temp[1:, :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo+startdpfileindex) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
        elif startdparrindex ==0 and enddparrindex >= 1:
            dp[avgFileSize-1:, :, :] = file_temp[:1, :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo+startdpfileindex-1) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
            dp[:, :, :] = file_temp[1:(avgFileSize+1), :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo+startdpfileindex) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
            dp[:enddparrindex, :, :] = file_temp[avgFileSize+1:, :, :]
        elif startdparrindex >=1 and enddparrindex == 0:
            dp[(startdparrindex - 1):(file_temp.shape[0]+startdparrindex - 1),:,:] = file_temp[:, :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo + startdpfileindex) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
        elif startdparrindex >=1 and enddparrindex >=1:
            dp[(startdparrindex - 1):avgFileSize, :, :] = file_temp[:(avgFileSize-startdparrindex+1), :, :]
            dp[dp < 0] = 0
            dp[dp > 1e7] = 0
            f = h5py.File(result_dir + 'fly' + '%4d' % (scanNo + startdpfileindex) + '_Ndp128_dp.h5', "w")
            f.create_dataset("dp", shape=dp.shape, dtype='float64', data=dp, compression="gzip")
            f.close()
            dp[:enddparrindex, :, :] = file_temp[(avgFileSize - startdparrindex + 1):, :, :]# restart from 0

    prefilesize = prefilesize + file_temp.shape[0];
