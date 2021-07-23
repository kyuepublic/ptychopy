import ptychopy
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import animation

str = "./ptycho -jobID=sim256rDM -algorithm=DM -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=100 -size=256 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1"
mlsstr= "./ptycho -jobID=IOTes256MLs5 -algorithm=MLs -fp=/home/beams/USER2IDD/ptychography/p2/ptycholib/scan152/scan152_data_#06d.h5 -fs=1 -hdf5path=/entry/data/data -beamSize=100e-6 \
         -qxy=276,616 -scanDims=51,51 -step=100e-9,100e-9 -i=30 -size=256 -lambda=1.408911284090909e-10 -dx_d=75e-6 -z=1.92 -dpf=51 -probeModes=2 -delta_p=0.1 -PPS=20"

# ptychopy.mlscmdstr(mlsstr)
# Step mode with command string
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ptychopy.mlsinit(mlsstr)
ptychopy.mlsstep()
data = ptychopy.mlsresobj()
type(data)
im = ax.imshow(np.angle(data), animated=True)
def update_image(i):
    for its in range(3):
        ptychopy.mlsstep()
    data = ptychopy.mlsresobj()
    im.set_array(np.angle(data))
    # time.sleep(1.5)
ani = animation.FuncAnimation(fig, update_image, frames=20, repeat=False)
plt.show()
ptychopy.mlspost()







