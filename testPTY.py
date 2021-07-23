import ptychopy
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import animation

str = "./ptycho -jobID=sim256rDM -algorithm=DM -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=100 -size=256 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1"
realstr= "./ptycho -jobID=IOTes256MLs -algorithm=MLs -fp=/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5 -fs=1 -hdf5path=/entry/data/data -beamSize=100e-6 \
         -qxy=276,616 -scanDims=51,51 -step=100e-9,100e-9 -i=30 -size=256 -lambda=1.408911284090909e-10 -dx_d=75e-6 -z=1.92 -dpf=51 -probeModes=2 -delta_p=0.1 -PPS=20"

l1 = np.array([1.1,2.1,3.1])
l2 = np.array([[1.0,2.0,3.0], [4.0,5.0,6.0], [7.0,8.0,9.0], [3.0, 5.0, 0.0]])
# l3 = np.array([[2,7, 1], [6, 3, 9], [1, 10, 13], [4, 2, 6]])  # Line A
# l3 = np.array([])                                             # Line B
l3 = np.array([[[2,7, 1, 11], [6, 3, 9, 12]],
                 [[1, 10, 13, 15], [4, 2, 6, 2]]])
l4 = np.array([[1.0+1.1j,2.0+1.6j,3.0+1.j], [4.0+1.j,5.0+1.j,6.0+1.j], [7.0+1.j,8.0+1.j,9.0+1.j], [3.0+1.j, 5.1+1.j, 0.0+1.j]])
# ptychopy.helloworld(l1, l2, l3)
# object = np.genfromtxt('data/sim512_object_0.csv', delimiter=',', dtype=complex)
# np.savetxt('testobject.csv',object, delimiter=',')
# Use one single API for calling
# ptychopy.epiecmdstr(str)
# ptychopy.dmcmdstr(str)
ptychopy.mlscmdstr(realstr)

# Whole mode with keyworkd
# ptychopy.epie(jobID="ePIEsimu128", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
#               stepy=50e-9, lambd=2.4796837508399954e-10, iter=100, size=128, dx_d=172e-6, z=1, simulate=1);
# ptychopy.dm(jobID="DMsimR512", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
#               stepy=50e-9, lambd=2.4796837508399954e-10, iter=30, size=512, dx_d=172e-6, z=1, simulate=1);
# ptychopy.mls(jobID="ePIEsimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
#               stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=0);


# Whole mode key word, with IO file as input
# ptychopy.epie(jobID="ePIEIOTestr256", fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
#               probeModes=1)
# ptychopy.dm(jobID="DMIOTestr256", fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
#               probeModes=1)

# ptychopy.mls(jobID="MLSIOTestr256", fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
#               probeModes=1, delta_p=0.1, PPS=20)

# Whole mode key work, with numpy array as input objectNP=l4,      positionNP=l2,   objectNP=l2, probeNP=l4,
# ptychopy.epienp(jobID="ePIEIOTestr256", diffractionNP=l3, fp="/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5", \
#              fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
#               stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92, dpf=51, \
#               probeModes=1)


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

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ob=np.genfromtxt(str("./data/ePIEsimu1_object_0.csv"),delimiter=',',dtype=complex)
# im = ax.imshow(np.angle(data), animated=True)
# plt.show()






