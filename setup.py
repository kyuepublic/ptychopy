try:
    from setuptools import setup, Extension, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension, find_packages
    from distutils.command.build_ext import build_ext


import numpy
import numpy.distutils.misc_util
import pkg_resources
import os
from os.path import join as pjoin
import platform
import sys


os.environ['CC'] = 'g++'
os.environ['CPP'] = 'g++'
os.environ['PTYCHOPY_BASE'] = os.getcwd()

# print("Setting env var PTYCHOPY_BASE to {:s}.".format(os.getcwd()))
# print(os.path.dirname(os.path.dirname(sys.executable)))

# virPythonPath = pjoin(os.path.dirname(os.path.dirname(sys.executable)), 'include')
virPythonPath = os.path.dirname(os.path.dirname(sys.executable))

def customize_compiler(self):

    self.src_extensions.append('.cu')

    try:
        self.compiler_so.remove('-Wstrict-prototypes')
    except ValueError:
        pass
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            try:
                postargs = extra_postargs['nvcc']
            except:
                postargs = [os.path.abspath('.')]
        elif os.path.splitext(src)[1] == '.c':
            postargs = extra_postargs['g++']
        elif os.path.splitext(src)[1] == '.cpp':
            self.compiler_so = default_compiler_so
            postargs = extra_postargs['g++']
        else:
            postargs = extra_postargs['g++']

        # else:
        #     postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler_so = default_compiler_so

    self._compile = _compile

class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        build_ext.build_extensions(self)

def locate_cuda():
    """
    Locate where cuda is installed on the system.

    Return a dict with values 'home', 'nvcc', 'sdk', 'include' and 'lib'
    and values giving the absolute path to each directory.

    Looks for the CUDAHOME environment variable. If not found, searches $PATH
    for 'cuda' and 'nvcc'.

    Adapted from:
    http://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code
    """
    home = None
    if platform.architecture()[0]=='64bit':
        arch = '64'
    else:
        arch = ''

    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise search PATH for cuda and nvcc
        for element in os.environ['PATH'].split(os.pathsep):
            if element.find('cuda')>-1:
                home = element.rstrip('/bin')
                nvcc = pjoin(home, 'bin', 'nvcc')
            elif element.find('nvcc')>-1:
                nvcc = os.path.abspath(element)
                home = os.path.dirname(os.path.dirname(element))
    if not home:
        raise EnvironmentError('The nvcc binary could not be located be '
                'located in your $PATH. Either add it to your path, or set '
                'the CUDAHOME environment variable.')
    cudaconfig = {'home':home, 'nvcc':nvcc,
            'sdk':pjoin(home, 'samples'),
            'include':pjoin(home, 'include'),
            'lib':pjoin(home, 'lib'+arch) }
    for key, val in cudaconfig.items():
        if not os.path.exists(val):
            raise EnvironmentError('The CUDA path {:s} could not be located in {:s}'.format(key, val))
    if 'CUDACOMPUTE' in os.environ:
        cudaconfig['sm'] = '{:d}'.format(int(10*float(os.environ['CUDACOMPUTE'])))
        cudaconfig['compute'] = '{:d}'.format(int(10*float(os.environ['CUDACOMPUTE'])))
    else:
        raise EnvironmentError("The 'CUDACOMPUTE' environment variable was "
                "was not set.")

    # print("-gencode=arch=compute_{:s},code=sm_{:s}".format(cudaconfig['compute'],cudaconfig['sm']))

    return cudaconfig

def locate_hdf5():
    """
    Locate where the HDF5 libraries are located on the system.

    Returns a dict with keys 'home', 'lib' and 'include'
    and values giving the absolute path to each directory.

    Looks for the HDF5HOME environment variable.
    """

    home = None
    if platform.architecture()[0]=='64bit':
        arch = '64'
    else:
        arch = ''
    if 'HDF5_BASE' in os.environ:
        home = os.environ['HDF5_BASE']
    else:
        raise EnvironmentError('Unable to locate the HDF libraries on '
                'this system. Set the HDF5_BASE environment variable.')

    hdfconfig = {'home':home,
            'lib':pjoin(home, 'lib'),
            'include':pjoin(home, 'include')}
    for key, val in hdfconfig.items():
        if not os.path.exists(val):
            raise EnvironmentError('Could not locate {:s} in path {:s}'.format(key, val))
    return hdfconfig

def locate_diy():
    """
    Locate where the DIY libraries are located on the system.

    Returns a dict with keys 'home', 'lib' and 'include'
    and values giving the absolute path to each directory.

    Looks for the DIYHOME environment variable.
    """
    home = None
    if platform.architecture()[0]=='64bit':
        arch = '64'
    else:
        arch = ''
    if 'DIY_BASE' in os.environ:
        home = os.environ['DIY_BASE']
    else:
        raise EnvironmentError('Unable to locate the DIY libraries on '
                'this system. Set the DIY_BASE environment variable.')

    # diyconfig = {'home':home,
    #         'lib':pjoin(home, 'lib'),
    #         'include':pjoin(home, 'include')}

    diyconfig = {'home':home,
            'include':pjoin(home, 'include')}

    for key, val in diyconfig.items():
        if not os.path.exists(val):
            raise EnvironmentError('Could not locate {:s} in path {:s}'.format(key, val))
    return diyconfig

CUDA = locate_cuda()
HDF5 = locate_hdf5()
# DIY = locate_diy()

if platform.architecture()[0]=='32bit':
    suffix = ''
else:
    suffix = '64'

with open('./requirements.txt','r') as f_requirements:
    requirements = f_requirements.readlines()

# Create ptychopy.so
extptychopy = Extension(name='ptychopy',
        sources=[
            './src/ScanMeshKernels.cu',
            './src/SampleKernels.cu',
            './src/ProbeKernels.cu',
            './src/utilitiesKernels.cu',
            './src/DiffractionsKernels.cu',
            './src/CartesianScanMesh.cpp',
            './src/CudaSmartPtr.cpp',
            './src/Cuda2DArray.cpp',
            # './src/GLResourceManager.cpp',
            # './src/RenderServer.cpp',
            './src/IPtychoScanMesh.cpp',
            './src/DiffractionLoader.cpp',
            './src/ThreadPool.cpp',
            './src/ListScanMesh.cpp',
            './src/Sample.cpp',
            './src/Probe.cpp',
            './src/Diffractions.cpp',
            './src/CudaFFTPlan.cpp',
            './src/utilities.cpp',
            './src/IPhaser.cpp',
            # './src/MPIPhaser.cpp',
            './src/ePIE.cpp',
            './src/MLS.cpp',
            './src/Parameters.cpp',
            './src/FileManager.cpp',
            './src/Timer.cpp',
            './src/SpiralScanMesh.cpp',
            './src/DM.cu',
            # './src/WS/easywsclient.cpp',
            './src/ptychopy.c',
            ],
        language='c++',
        include_dirs=[
            numpy.get_include(),
            numpy.distutils.misc_util.get_numpy_include_dirs(),
            CUDA['include'],
            CUDA['sdk']+'/common/inc',
            # '/local/kyue/program/anaconda/envs/py36/include',
            virPythonPath + '/include',
            './src',
            # HDF5['include'],
            # '/local/kyue/anlproject/ptography/githubptychopy/ptychopy/src',
            ],
        library_dirs=[
            CUDA['lib'],
            CUDA['sdk']+'/common/lib',
            # HDF5['lib'],
            # './src/Eigen/src',
            ],
        runtime_library_dirs=[CUDA['lib']],
        extra_link_args = ['-lcudart', '-lcurand', '-lcufft', '-lcublas', '-lhdf5', '-lhdf5_cpp', '-lpthread'],
        extra_compile_args={
            'nvcc': ['-Xcompiler', '-fpic', '-O3', '-gencode=arch=compute_{:s},code=sm_{:s}'.format(CUDA['compute'],CUDA['sm'])],
            'gcc': ['-DEPIE_HDF5',  '-fpic', '-DHAVE_HDF5'],
            'g++': ['-DEPIE_HDF5',  '-fpic', '-DHAVE_HDF5']
            })

        # extra_link_args = ['-lcudart', '-lcurand', '-lcufft', '-lcublas', '-lSDL2', '-lGLEW', '-lGL', '-lhdf5', '-lhdf5_cpp', '-lz', '-lsz', '-lpthread'],
        # extra_compile_args={
        #     'nvcc': ['-Xcompiler', '-fpic', '-O3','-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_{:s},code=sm_{:s}'.format(CUDA['compute'],CUDA['sm'])],
        #     'gcc': ['-DEPIE_HDF5', '-DHAVE_DIY'],
        #     'g++': ['-DEPIE_HDF5', '-DHAVE_DIY'],
        #     'clang': ['-DEPIE_HDF5', '-DHAVE_DIY'],
        #     'clang++': ['-DEPIE_HDF5', '-DHAVE_DIY'],
        #     }

config = {
    'name': 'ptychopy',
    'version': open('VERSION').read().strip(),

   # 'packages': ['ptychopy'],
    'include_package_data': True,
    'description': 'Fast ptychography reconstruction library.',

    'ext_modules': [extptychopy],
    'cmdclass':{'build_ext':custom_build_ext},

    'author': 'Ke Yue, Junjing Deng, David J. Vine',
    'author_email': 'kyue@anl.gov, junjingdeng@anl.gov, djvine@gmail.com',

    # 'url': 'http://aps.anl.gov/ptychopy',
    'download_url': 'https://github.com/kyuepublic/ptychopy.git',
    'install_requires': requirements,

    'license': 'BSD',
    'platforms': 'Any',

    'classifiers': ['Development Status :: 1 - Pre-alpha',
        'Licence :: OSI Approved :: BSD Licence',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        # 'Programming Language :: Python 2.7',
        'Programming Language :: Python 3.4',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Cuda',
        'Topic :: Scientific/Engineering :: Physics',
        ]
}

setup(**config)
