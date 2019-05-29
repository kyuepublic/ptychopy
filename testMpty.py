import ptychopy


str = "./ptycho -jobID=sim512 -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=3 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0"


import matplotlib.pyplot as plt

print(str)
ptychopy.epieinit(str)

for i in range(3):
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