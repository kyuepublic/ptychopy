#import ptychopy


# str = "./ptycho -jobID=sim512 -algorithm=ePIE -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=3 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1"
# realstr= "./ptycho -jobID=IOTest -fp=/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5 -fs=1 -hdf5path=/entry/data/data -beamSize=100e-6 -algorithm=MLs -qxy=276,616 -scanDims=51,51 -step=100e-9,100e-9 -i=3 -size=256 -lambda=1.408911284090909e-10 -dx_d=75e-6 -z=1.92 -dx=1.408911284090909E-8 -dpf=51 -probeModes=5 -delta_p=0.1 -PPS=20"

import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import animation

# Use one single API for calling
# ptychopy.epiecmdstr(str)
# ptychopy.dmcmdstr(str)
# ptychopy.mlscmdstr(realstr)

# Whole mode with keyworkd
# ptychopy.epie(jobID="ePIEsimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
#               stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=1);
#ptychopy.dm(jobID="ePIEsimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
              #stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=1);
# ptychopy.mls(jobID="ePIEsimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
#               stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=0);

# Step mode with command string
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
#
# ptychopy.epieinit(str)
#
# ptychopy.epiestep()
#
# data = ptychopy.epieresobj()
# type(data)
# im = ax.imshow(np.angle(data), animated=True)
#
# def update_image(i):
#     for its in range(3):
#         ptychopy.epiestep()
#     data = ptychopy.epieresobj()
#     im.set_array(np.angle(data))
#     # time.sleep(1.5)
#
# ani = animation.FuncAnimation(fig, update_image, frames=4, repeat=False)
#
# plt.show()
#
# ptychopy.epiepost()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ob=np.genfromtxt(str("./data/ePIEsimu1_object_0.csv"),delimiter=',',dtype=complex)
im = ax.imshow(np.angle(data), animated=True)
plt.show()






