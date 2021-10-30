# ptychopy
This software package has a backend implemented with CUDA C++ and MPI, a frontend implemented with CPython numpy
interface to have a high performance python interface for ptychography image reconstruction. The CUDA C++ backend provides
a faster solution compared to python implementation in terms of reconstruction speed. It could be run on either single GPU, 
or multiple GPUs on supercomputer. It could be used as either a C++ binarary or python package.

## Quickstart

1. Need to install python3 to run the GUI and ptychopy, other needed library is
   in requirement.txt.(Tested OS RHEL 6.0, 7.0). This library could also be
   compiled as a CUDA-C library. Inside src folder, change build.sh with your
   HDF library path.

2. Recommend conda virtual environment, for example

```
conda create -n py36 python=3.6 hdf5-external-filter-plugins-lz4
```

3. Activate the virtual environment

```
source activate py36

```

4. To install and build the python package, set environment variables HDF5_BASE
   and CUDAHOME, which point to the installed path of the HDF5 and CUDA
   libraries. Also, set the cuda computing based on your GPU. 
   
   Recommend CUDA version <=10. For latest cuda version, you might run into some
   compiliation issue and need to do some tricks to link to other libraryes.
   
   For example the
   2080 Ti has compute capability 7.5. The GPU computing capability number can
   be found on the [NVidia website](https://developer.nvidia.com/cuda-gpus)

```
export CUDACOMPUTE=7.5
export CUDAHOME=/user/local/cuda-8.0
export HDF5_BASE=$CONDA_PREFIX

```

5. Then just call the following

```
./install.sh

```

6. For testing, you can use epie.py or mls.py

```
python epie.py

```

## Description

This software package is implemented with ePIE, DM, MLs, reconstruction
algorithm. It supports list, spiral, cartesian, scan position file. To load a
predefined scan position file, object value files, probe value files, use the
following parameters to do the reconstruction.

## Python API

There are two sets API for doing the reonstrunction, one set is for the whole
mode and the other set is for step mode.  The step mode is used mostly for
debug purpose, so the result for each reconstruction step could be checked. And
the whole mode would finish the whole reconstruction process and write the
result to the disk. The step mode would be slower than the whole mode, since
each step would need to transfer data back from GPU to CPU side.

The following example is based on the simulation data. For real data, the -fp
parameter has to be specified. The file format must be a bunch of HDF5 files
and the file path would be like filename_data_#06d.h5. The library will go
though all the HDF5 files and load the diffraction pattern into the library.

Whole mode API example with simulation images:
    
=======
```
Use simulation image as the example input
    ptychopy.epie(jobID="epiesimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
               stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=1);

    ptychopy.dm(jobID="dmsimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
              stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=1);

    ptychopy.mls(jobID="mlssimu1", beamSize=110e-9, scanDimsx=30, scanDimsy=30, stepx=50e-9, \
               stepy=50e-9, lambd=2.4796837508399954e-10, iter=3, size=512, dx_d=172e-6, z=1, simulate=0);

Use real data as the example input
    ptychopy.epie(jobID="ePIEIOTestr256", fp="/home/scan152/scan152_data_#06d.h5", \
                 fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
                  probeModes=5)

    ptychopy.dm(jobID="DMIOTestr256", fp="/home/scan152/scan152_data_#06d.h5", \
                 fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
                  probeModes=1)
    
    ptychopy.mls(jobID="MLSIOTestr256", fp="/home/scan152/scan152_data_#06d.h5", \
                 fs=1, hdf5path="/entry/data/data", beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=100, size=256, dx_d=75e-6, z=1.92, dpf=51, \
                  probeModes=1, delta_p=0.1, PPS=20)

```

Pass diffraction pattern as a 3D numpy array, diffractionNP, objectNP, probeNP
```
    ptychopy.epienp(jobID="ePIEIOTestr256", diffractionNP=dp,\
                 fs=1, beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92,\
                  probeModes=2)
    
    ptychopy.dmnp(jobID="dmIOTestr256", diffractionNP=dp,\
                 fs=1, beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92,\
                  probeModes=2)
    
    ptychopy.mlsnp(jobID="mlsIOTestr256", diffractionNP=dp,\
                 fs=1, beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92, \
                    probeModes=2)

```

Step mode API example:

```

    epienpinit(jobID="ePIEIOTestr256", diffractionNP=dp,\
                 fs=1, beamSize=110e-6, qx=276, qy=616, scanDimsx=51, scanDimsy=51, stepx=100e-9, \
                  stepy=100e-9, lambd=1.408911284090909e-10, iter=10, size=256, dx_d=75e-6, z=1.92,\
                  probeModes=2)
    epiestep()
    episresobj()
    epiresprobe()
    epipost()

```

For implentation example, please check example folder.

## C++ binary 
To use it as C++ binary, first go to src folder and change the correspondnig hdf5 path,
after compiling, use the following example command for a test simulation image:

```
./ptycho -jobID=sim512ePIE -algorithm=ePIE -beamSize=110e-9 -scanDims=30,30 
-step=50e-9,50e-9 -i=100 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1
```

To use it on the real data, first you have to have a bounch of hdf5 files which have all
the diffraction pattern datas. For the real example, -hdf5path=/entry/data/data means the
diffraction pattern data is saved under /entry/data/data, and each file has -dpf=51 or 51 diffraction patterns.

```
./ptycho -jobID=IOTest512ePIE -algorithm=ePIE -fp=/data2/scan152/scan152_data_#06d.h5 -fs=1 
-hdf5path=/entry/data/data -beamSize=100e-6  -qxy=276,616 -scanDims=51,51 -step=100e-9,100e-9 
-i=100 -size=512 -lambda=1.408911284090909e-10 -dx_d=75e-6 -z=1.92 -dpf=51 -probeModes=2
```

To use it on super computer, the DIY path has to be set, the DIY library is included in the code.
The MPI library has to installed and set. Currently only ePIE and DM algorithm is supported. 
A example to launch 2 MPI processes:

```
mpiexec -n 2 ./ptycho -jobID=sim512c -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=20 
-size=512  -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1
```

## Parameters

  Name | Type | Description | Default
:------------: | :-------------: | :------------: | :------------:
jobID | string  | An identifying tag to the reconstruction run | ``
algorithm | string  | The algorithm to use for reconstruction. Accepted values ( `ePIE,DM, MLs` ) | `ePIE`
fp | path  | A c-style formatted string for the location of the HDF5 files. For file name string substitution starting at `fs` . Example: `-fp=/data/diff_#03d.h5` for files in the form `diff_000.h5, diff_001.h5, ...` | `N/A`
fs | integer  | The file index of the file containing the first diffraction pattern (top left corner for Cartesian scans) | `0`
hdf5path | string | Diffraction data HDF5 dataset name | `/entry/data/data`
dpf | integer  | The number of diffraction patterns per file | `1`
beamSize | real  | The theoretical incident beam size in meters | `110e-9`
probeGuess | path  | The location of a CSV complex valued file used as the initial probe guess for reconstruction (The file can be saved with python using np.savetxt(numpyProbe, delimiter=', ') and it has to have the same dimensions as the `size` parameter | `N/A`
objectGuess | path  | The location of a CSV complex valued file used as the initial object guess for reconstruction | `N/A`
size | integer  | The desired size for cropping the diffraction patterns and probe size. Preferably a multiple of 2: 128, 256, 512, etc... | `256`
qxy | integer, integer  | The center of the diffraction pattern in pixels (image pixel location Y, image pixel location X). Diffraction patterns will be cropped to a square image of sizeXsize pixels around qxy. | `128, 128`
nxy | integer, integer  | The size of the diffraction pattern before pre-processing (rows, columns). Only required for .csv and .bin files; the dataset dimensions can be detected automatically in HDF5 files. | `256, 256`
scanDims | integer, integer  | The grid dimensions for Cartesian scans (rows, columns) | `26, 26`
spiralScan | (0, 1)  | Use a spiral scan mesh instead of a Cartesian grid | `0`
flipScanAxis | (0, 1)  | Flips the raster scan direction from horizontal to vertical | `0`
mirror1stScanAxis | (0, 1)  | Flips the raster scan direction along the first axis (vertically, downwards to upwards, if flipScanAxis=0) | `0`
mirror2ndScanAxis | (0, 1)  | Flips the raster scan direction along the second axis (horizontally, left-to-right to right-to-left, if flipScanAxis=0) | `0`
step | real, real  | The step size in meters for each of the scan grid dimensions (row step, column step) | `40e-9, 40e-9`
probeModes | integer  | Number of orthogonal probe modes to simulate partial incoherence of the beam | `1`
lambda | real  | Wavelength of the incident beam in meters (calculated from the energy used) | `2.3843e-10`
dx_d | real  | Detector pixel size in meters | `172e-6`
z | real  | Distance between sample and detector in meters | `2.2`
iter | integer  | Number of reconstruction iterations | `100`
T | integer  | Maximum allowed reconstruction time (in sec). Overrides iterations. | `N/A`
jitterRadius | integer  | Radius in pixels for random displacement of raster scan positions | `0`
delta_p | real | LSQ damping constant, used only for MLs method | 0.1
threshold | integer  | To remove noise from the diffraction patterns. Any count below this number will be set to zero in the diffraction data. | `0`
rotate90 | (0, 1, 2, 3)  | Number of times to rotate the diffraction patterns by 90 degrees. (Currently only a value of 1 is supported) | `0`
sqrtData | flag  | Take the sqrt of the loaded diffraction pattern magnitudes | `N/A`
fftShiftData | flag  | Apply an fftshift operation to the diffraction patterns | `N/A`
blind | (0, 1)  | Turn visualization on(0)/off(1). Only has an effect when the library is built with SDL support or running from the GUI | `1`
binaryOutput | (0, 1)  | Write results in binary format (1), or CSV format (0) | `0`
simulate | (0, 1)  | Run a test simulation from lena and baboon images as magnitude and phase profiles of a synthetic sample wavefront. (Only works with -size=128, 256, 512, 1024) | `0`
overlap | integer  | Only has an effect when the library is built with DIY support and is running on multiple GPUs. The size of a halo in raster scan dimensions where the data is shared between multiple reconstructions running on multiple GPUs | `0`
shareFrequency | integer  | Only has an effect when the library is built with DIY support and is running on multiple GPUs. Determines the frequency of data sharing among GPUs, in terms of number of iterations | `10`
phaseConstraint | integer  | The number of iterations to keep applying a phase constraint (forcing the reconstructed phase in the range [-2pi, 0]) | `1`
updateProbe | integer  | The number of iterations after which to start updating the primary probe mode | `10`
updateModes | integer  | The number of iterations after which to start updating all probe modes | `20`
updateVis | integer  | The number of iterations after which to start updating the visualization. Only has an effect when the library is built with SDL support or running from the GUI | `10`
beamstopMask | (0, 1)  | Determine whether the beamstop area (0 values, set by a binary array "beamstopMask.h5" which is put in the data directory) is applied with Fourier modulus constraint or not  | `0`
lf | path  | The location of a CSV complex valued file for positions (The file can be saved with python using np.savetxt(numpyProbe, delimiter=', ') and it has to be arranged with position y, x in each row, the unit is m  | `N/A`
PPS | integer  | The number of iterations after which to start probe position search, only work for MLs method | `20`


  Name | Type | Description | Default
:------------: | :-------------: | :------------: | :------------:
diffractionNP | real  | numpy array for diffraction | `numpy array for diffraction`
objectNP | complex  | numpy array for object array | `numpy array for object array`
probeNP | complex  | numpy array for probe array | `numpy array for probe array`

## Reference Paper

```
@inproceedings{yue2021ptychopy,
  title={Ptychopy: GPU framework for ptychographic data analysis},
  author={Yue, Ke and Deng, Junjing and Jiang, Yi and Nashed, Youssef and Vine, David}
}

```
