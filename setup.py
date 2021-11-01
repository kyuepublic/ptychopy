from skbuild import setup

with open('./requirements.txt', 'r') as f_requirements:
    requirements = f_requirements.readlines()

setup(
    name='ptychopy',
    version=open('VERSION').read().strip(),
    packages=['ptychopy'],
    package_dir={"": "src"},
    zip_safe=False,
    description='Fast ptychography reconstruction library.',
    author='Ke Yue, Junjing Deng, David J. Vine',
    author_email='kyue@anl.gov, junjingdeng@anl.gov, djvine@gmail.com',
    download_url='https://github.com/kyuepublic/ptychopy.git',
    install_requires=requirements,
    license='BSD',
    platforms='Any',
    classifiers=[
        'Development Status :: 1 - Pre-alpha',
        'Licence :: OSI Approved :: BSD Licence',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python 3.5',
        'Programming Language :: C',
        'Programming Language :: C++',
        'Programming Language :: Cuda',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
