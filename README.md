# ptychopy

1 Need to install python3 to run the GUI and ptychopy, other needed library is in requirement.txt.(Tested OS RHEL 6.0, 7.0)

2 Recommend conda virtual environment, for example

  conda create -n py34 python=3.4

3 Activate the virtual environment

  source activate py34

4 Set the cuda computing based on your GPU, for example

  export CUDACOMPUTE=7.5 (2080ti)
  The GPU computing capability number can be found on the nvidia website

5 To install and build the python package, set another two environment varialbe
  HDF5_BASE and CUDAHOME, which point to the installed path of the CUDA and HDF5
  library. Then just call the following
  
  ./install.sh

6 For testing, you can use testPTY.py or testMpty.py

  python testPTY.py
  


# parameters
  
  Name | Type | Description | Default
:------------: | :-------------: | :------------: | :------------:
jobID | string  | An identifying tag to the reconstruction run | ``
algorithm | string  | The algorithm to use for reconstruction. Accepted values (`ePIE,DM`) | `ePIE`
fp | path  | A c-style formatted string for the location of the HDF5 files. For file name string substitution starting at `fs`. Example: `-fp=/data/diff_#03d.h5` for files in the form `diff_000.h5, diff_001.h5, ...` | `N/A`
fs | integer  | The file index of the file containing the first diffraction pattern (top left corner for Cartesian scans) | `0`
hdf5path | string | Diffraction data HDF5 dataset name | `/entry/data/data`
dpf | integer  | The number of diffraction patterns per file | `1`
beamSize | real  | The theoretical incident beam size in meters | `110e-9`
probeGuess | path  | The location of a CSV complex valued file used as the initial probe guess for reconstruction (The file can be saved with python using np.savetxt(numpyProbe, delimiter=',') and it has to have the same dimensions as the `size` parameter | `N/A`
objectGuess | path  | The location of a CSV complex valued file used as the initial object guess for reconstruction | `N/A`
size | integer  | The desired size for cropping the diffraction patterns and probe size. Preferably a multiple of 2: 128,256,512,etc... | `256`
qxy | integer,integer  | The center of the diffraction pattern in pixels (image pixel location Y, image pixel location X). Diffraction patterns will be cropped to a square image of sizeXsize pixels around qxy. | `128,128`
nxy | integer,integer  | The size of the diffraction pattern before pre-processing (rows, columns). Only required for .csv and .bin files; the dataset dimensions can be detected automatically in HDF5 files. | `256,256`
scanDims | integer,integer  | The grid dimensions for Cartesian scans (rows, columns) | `26,26`
spiralScan | (0,1)  | Use a spiral scan mesh instead of a Cartesian grid | `0`
flipScanAxis | (0,1)  | Flips the raster scan direction from horizontal to vertical | `0`
mirror1stScanAxis | (0,1)  | Flips the raster scan direction along the first axis (vertically, downwards to upwards, if flipScanAxis=0) | `0`
mirror2ndScanAxis | (0,1)  | Flips the raster scan direction along the second axis (horizontally, left-to-right to right-to-left, if flipScanAxis=0) | `0`
step | real,real  | The step size in meters for each of the scan grid dimensions (row step, column step) | `40e-9,40e-9`
probeModes | integer  | Number of orthogonal probe modes to simulate partial incoherence of the beam | `1`
lambda | real  | Wavelength of the incident beam in meters (calculated from the energy used) | `2.3843e-10`
dx_d | real  | Detector pixel size in meters | `172e-6`
z | real  | Distance between sample and detector in meters | `2.2`
i | integer  | Number of reconstruction iterations | `100`
T | integer  | Maximum allowed reconstruction time (in sec). Overrides iterations. | `N/A`
jitterRadius | integer  | Radius in pixels for random displacement of raster scan positions | `0`
threshold | integer  | To remove noise from the diffraction patterns. Any count below this number will be set to zero in the diffraction data. | `0`
rotate90 | (0,1,2,3)  | Number of times to rotate the diffraction patterns by 90 degrees. (Currently only a value of 1 is supported) | `0`
sqrtData | flag  | Take the sqrt of the loaded diffraction pattern magnitudes | `N/A`
fftShiftData | flag  | Apply an fftshift operation to the diffraction patterns | `N/A`
blind | (0,1)  | Turn visualization on(0)/off(1). Only has an effect when the library is built with SDL support or running from the GUI | `1`
binaryOutput | (0,1)  | Write results in binary format (1), or CSV format (0) | `0`
simulate | (0,1)  | Run a test simulation from lena and baboon images as magnitude and phase profiles of a synthetic sample wavefront. (Only works with -size=128,256,512,1024) | `0`
overlap | integer  | Only has an effect when the library is built with DIY support and is running on multiple GPUs. The size of a halo in raster scan dimensions where the data is shared between multiple reconstructions running on multiple GPUs | `0`
shareFrequency | integer  | Only has an effect when the library is built with DIY support and is running on multiple GPUs. Determines the frequency of data sharing among GPUs, in terms of number of iterations | `10`
phaseConstraint | integer  | The number of iterations to keep applying a phase constraint (forcing the reconstructed phase in the range [-2pi,0]) | `1`
updateProbe | integer  | The number of iterations after which to start updating the primary probe mode | `10`
updateModes | integer  | The number of iterations after which to start updating all probe modes | `20`
updateVis | integer  | The number of iterations after which to start updating the visualization. Only has an effect when the library is built with SDL support or running from the GUI | `10`
beamstopMask | (0,1)  | Determine whether the beamstop area (0 values, set by a binary array "beamstopMask.h5" which is put in the data directory) is applied with Fourier modulus constraint or not  | `0`
lf | path  | The location of a CSV complex valued file for positions (The file can be saved with python using np.savetxt(numpyProbe, delimiter=',') and it has to be arranged with position y, x in each row, the unit is m  | `N/A`
