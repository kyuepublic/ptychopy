import ptychopy


# str = "./ptycho -jobID=sim512 -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=3 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0"

# str = "./ptycho jobID=IC_8.800keV_fly145_p10_s256_ts1_i50 -fp=/data2/JunjingData/fly145/data_roi0_dp_#d.h5 -fs=1 \
#      -beamSize=400.0e-9 -qxy=282.0,663.0 -scanDims=1,1800 -probeModes=10 -i=50 -rotate90=0 -sqrtData -fftShiftData \
#      -threshold=1 -size=256 -lambda=0.140891e-9 -dx_d=75e-6 -z=1.920 -dpf=1800 -hdf5path=/dp -probeGuess= -updateProbe=20 -updateModes=30 lf=/data2/JunjingData/fly145/fly145_pos.csv"

str = "./ptycho jobID=IC_8.800keV_fly145_p10_s256_ts1_i50 -fp=/data2/JunjingData/mlhcode/fly145/fly145_data_#06d.h5 -fs=1 -beamSize=400.0e-9 \
    -qxy=282.0,663.0 -scanDims=36,50 -probeModes=10 -i=50 -rotate90=0 -sqrtData -fftShiftData -threshold=1 -size=256 -lambda=0.140891e-9 \
     -dx_d=75e-6 -z=1.920 -dpf=50 -hdf5path=/entry/data/data -probeGuess= -updateProbe=20 -updateModes=30 lf=/data2/JunjingData/mlhcode/fly145/fly145_pos.csv"

import matplotlib.pyplot as plt

print(str)
ptychopy.epieinit(str)

for i in range(50):
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