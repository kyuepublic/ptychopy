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
