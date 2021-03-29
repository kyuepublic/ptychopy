import ptychopy


str = "./ptycho -jobID=sim512 -algorithm=ePIE -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=60 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0"


import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import animation

# Use one single API for calling
ptychopy.epiecmdstr(str)
# ptychopy.epie(iter=100);
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





