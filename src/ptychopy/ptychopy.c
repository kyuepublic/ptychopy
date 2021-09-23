////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Copyright © 2019, UChicago Argonne, LLC
//
//All Rights Reserved
//
//Software Name: ptychopy
//
//By: Argonne National Laboratory
//
//OPEN SOURCE LICENSE
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
//following conditions are met:
//
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
//disclaimer in the documentation and/or other materials provided with the distribution.
//
//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
//derived from this software without specific prior written permission.
//
//DISCLAIMER
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
//WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex.h>
//#include "SimplePhaser.h"
#include "MPIPhaser.h"
#include "Parameters.h"
#include "utilities.h"
#include "Cuda2DArray.hpp"
#include "CudaSmartPtr.h"
#include "Sample.h"
#include "Probe.h"

#include <iostream>

#define PY_ARRAY_UNIQUE_SYMBOL

using namespace std;

/* Define module and function docstrings */
static char module_docstring[] = "This module provides an interface to ptycholib GPU-accelerated phase retrieval.";
static char helloworld_docstring[] = "This function exists for debugging and testing purposes.";

static char epie_docstring[] = "This function performs iterative phase retrieval using the ePIE algorithm with keyword.";
static char dm_docstring[] = "This function performs iterative phase retrieval using the dm algorithm with keyword.";
static char mls_docstring[] = "This function performs iterative phase retrieval using the mls algorithm with keyword.";

static char epiecmdstr_docstring[] = "This function performs iterative phase retrieval using the ePIE algorithm \
            with a whole string.";
static char dmcmdstr_docstring[] = "This function performs iterative phase retrieval using the DM algorithm \
            with a whole string.";
static char mlscmdstr_docstring[] = "This function performs iterative phase retrieval using the MLs algorithm \
            with a whole string.";

static char epieinit_docstring[] = "This function performs the init.";
static char epiepost_docstring[] = "This function performs the post.";
static char epiestep_docstring[] = "This function performs the step.";
static char epieresobj_docstring[] = "This function returns the object of the reconstruction.";
static char epieresprobe_docstring[] = "This function returns the probe of the reconstruction.";

static char mlsinit_docstring[] = "This function performs the init.";
static char mlspost_docstring[] = "This function performs the post.";
static char mlsstep_docstring[] = "This function performs the step.";
static char mlsresobj_docstring[] = "This function returns the object of the reconstruction.";
static char mlsresprobe_docstring[] = "This function returns the probe of the reconstruction.";

/* Define function headers */
static PyObject *ptycholib_helloworld(PyObject *self, PyObject *args);
static PyObject *ptycholib_epie(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_dm(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mls(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_epienp(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_dmnp(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlsnp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_epiecmdstr(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_dmcmdstr(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlscmdstr(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_epieinit(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epiepost(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epiestep(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epieresobj(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epieresprobe(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_epienpinit(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_mlsinit(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlspost(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlsstep(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlsresobj(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_mlsresprobe(PyObject *self, PyObject *args, PyObject *keywds);

static IPhaser* phaser;
static unsigned int istep = 0;

/* Define module methods */
static PyMethodDef module_methods[] = {
    {"helloworld", (PyCFunction)ptycholib_helloworld, METH_VARARGS, helloworld_docstring},
    {"epie", (PyCFunction)ptycholib_epie, METH_VARARGS|METH_KEYWORDS, epie_docstring},
    {"dm", (PyCFunction)ptycholib_dm, METH_VARARGS|METH_KEYWORDS, dm_docstring},
    {"mls", (PyCFunction)ptycholib_mls, METH_VARARGS|METH_KEYWORDS, mls_docstring},
    {"epienp", (PyCFunction)ptycholib_epienp, METH_VARARGS|METH_KEYWORDS, epie_docstring},
    {"dmnp", (PyCFunction)ptycholib_dmnp, METH_VARARGS|METH_KEYWORDS, dm_docstring},
    {"mlsnp", (PyCFunction)ptycholib_mlsnp, METH_VARARGS|METH_KEYWORDS, mls_docstring},
    {"epiecmdstr", (PyCFunction)ptycholib_epiecmdstr, METH_VARARGS|METH_KEYWORDS, epiecmdstr_docstring},
    {"dmcmdstr", (PyCFunction)ptycholib_dmcmdstr, METH_VARARGS|METH_KEYWORDS, dmcmdstr_docstring},
    {"mlscmdstr", (PyCFunction)ptycholib_mlscmdstr, METH_VARARGS|METH_KEYWORDS, mlscmdstr_docstring},
    {"epieinit", (PyCFunction)ptycholib_epieinit, METH_VARARGS|METH_KEYWORDS, epieinit_docstring},
    {"epiepost", (PyCFunction)ptycholib_epiepost, METH_VARARGS|METH_KEYWORDS, epiepost_docstring},
    {"epiestep", (PyCFunction)ptycholib_epiestep, METH_VARARGS|METH_KEYWORDS, epiestep_docstring},
    {"epieresobj", (PyCFunction)ptycholib_epieresobj, METH_VARARGS|METH_KEYWORDS, epieresobj_docstring},
    {"epieresprobe", (PyCFunction)ptycholib_epieresprobe, METH_VARARGS|METH_KEYWORDS, epieresprobe_docstring},
    {"epienpinit", (PyCFunction)ptycholib_epienpinit, METH_VARARGS|METH_KEYWORDS, epieinit_docstring},
    {"mlsinit", (PyCFunction)ptycholib_mlsinit, METH_VARARGS|METH_KEYWORDS, mlsinit_docstring},
    {"mlspost", (PyCFunction)ptycholib_mlspost, METH_VARARGS|METH_KEYWORDS, mlspost_docstring},
    {"mlsstep", (PyCFunction)ptycholib_mlsstep, METH_VARARGS|METH_KEYWORDS, mlsstep_docstring},
    {"mlsresobj", (PyCFunction)ptycholib_mlsresobj, METH_VARARGS|METH_KEYWORDS, mlsresobj_docstring},
    {"mlsresprobe", (PyCFunction)ptycholib_mlsresprobe, METH_VARARGS|METH_KEYWORDS, mlsresprobe_docstring},
    {NULL, NULL, 0, NULL}
};

/* Define the module and it's methods */
static struct PyModuleDef ptycholib_module = {
    PyModuleDef_HEAD_INIT,
    "ptychopy", /* Name of module */
    module_docstring, /* Module docstring */
    -1, /* Size of per-interpreter state of the module or -1 if module keeps state in global variables*/
    module_methods
};

/* Init the module */
PyMODINIT_FUNC PyInit_ptychopy(void)
{
    PyObject *m = PyModule_Create(&ptycholib_module);

    if (m==NULL)
        return NULL;
    /* Load numpy functionality */
    import_array();

    return m;
}

/* Function definitions */
static PyObject *ptycholib_helloworld(PyObject *self, PyObject *args)
{

//    if (!PyArg_ParseTuple(args, ""))
//        return NULL;
//    printf("Hello, World!");
    PyObject *list1_obj;
    PyObject *list2_obj;
    PyObject *list3_obj;
    if (!PyArg_ParseTuple(args, "OOO", &list1_obj, &list2_obj, &list3_obj))
        return NULL;

    double *list1;
    double **list2;
    double ***list3;

    //Create C arrays from numpy objects:
    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];
    if (PyArray_AsCArray(&list1_obj, (void *)&list1, dims, 1, descr) < 0 || PyArray_AsCArray(&list2_obj, (void **)&list2, dims, 2, descr) < 0 || PyArray_AsCArray(&list3_obj, (void ***)&list3, dims, 3, descr) < 0)
    {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }
    printf("1D: %f, 2D: %f, 3D: %f.\n", list1[2], list2[3][1], list3[1][0][2]);


    Py_RETURN_NONE;
}

static void ptychopy_algorithm(PyObject *args, PyObject *keywds, char *algorithm)
{
    // Default value for the algorithm parameters
    char *jobID="";
    char *fp="";
    int fs=0;
    char* hdf5path="";
    int dpf = 1;
    double beamSize=400e-9;
    char *probeGuess="";
    char *objectGuess="";
    int size=512;
    int qx=128;
    int qy=128;
    int nx=256;
    int ny=256;
    int scanDimsx=26;
    int scanDimsy=26;
    int spiralScan=0;
    int flipScanAxis=0;
    int mirror1stScanAxis=0;
    int mirror2ndScanAxis=0;
    double stepx=40e-9;
    double stepy=40e-9;
    int probeModes=1;
    double lambd=2.3843e-10;
    double dx_d=172e-6;
    double z=2.2;
    int iter=100;
    int T=0;
    int jitterRadius=0;
    double delta_p=0.1;
    int threshold=0;
    int rotate90=0;
    int sqrtData=0;
    int fftShiftData=0;
    int binaryOutput=0;
    int simulate=0;
    int phaseConstraint=1;
    int updateProbe=10;
    int updateModes=20;
    int beamstopMask=0;
    char *lf="";
    int PPS=20;

    static char *kwlist[] = {"jobID", "fp", "fs", "hdf5path", "dpf", "beamSize", "probeGuess", "objectGuess", \
    "size", "qx", "qy", "nx", "ny", "scanDimsx", "scanDimsy", "spiralScan", "flipScanAxis", "mirror1stScanAxis", \
    "mirror2ndScanAxis", "stepx", "stepy", "probeModes", "lambd", "dx_d", "z", "iter", "T", "jitterRadius", \
    "delta_p",  "threshold", "rotate90", "sqrtData", "fftShiftData", "binaryOutput", "simulate", \
    "phaseConstraint", "updateProbe", "updateModes", "beamstopMask", "lf", "PPS", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|ssisidssiiiiiiiiiiiddidddiiidiiiiiiiiiisi", kwlist,
                &jobID, &fp, &fs, &hdf5path, &dpf, &beamSize, &probeGuess, &objectGuess, \
                &size, &qx, &qy, &nx, &ny, &scanDimsx, &scanDimsy, &spiralScan, &flipScanAxis, &mirror1stScanAxis, \
                &mirror2ndScanAxis, &stepx, &stepy, &probeModes, &lambd, &dx_d, &z, &iter, &T, &jitterRadius, \
                &delta_p,  &threshold, &rotate90, &sqrtData, &fftShiftData, &binaryOutput, &simulate, \
                &phaseConstraint, &updateProbe, &updateModes, &beamstopMask, &lf, &PPS))

    printf("Running algorithm %s. \n", algorithm);
//    printf("the jobID is %s \n", jobID);
////    printf("the fp is %s \n", fp);
//    printf("the beamSize is %.8e \n", beamSize);
//    printf("the iter is %d \n", iter);
//    printf("the scanDims is %d, %d, stepx is %.8e, stepy is %.8e, size is %d, dx_d is %.8e, z is %.8e  \n", scanDimsx, scanDimsy, stepx, stepy, size, dx_d, z);
//    printf("the lambd is %.8e \n", lambd);
//    printf("the algorithm is %s, the simulate is %d \n", algorithm, simulate);

    CXParams::getInstance()->parseFromCPython(jobID, algorithm, fp, fs, hdf5path, dpf, beamSize, probeGuess, objectGuess, \
                size, qx, qy, nx, ny, scanDimsx, scanDimsy, spiralScan, flipScanAxis, mirror1stScanAxis, \
                mirror2ndScanAxis, stepx, stepy, probeModes, lambd, dx_d, z, iter, T, jitterRadius, \
                delta_p,  threshold, rotate90, sqrtData, fftShiftData, binaryOutput, simulate, \
                phaseConstraint, updateProbe, updateModes, beamstopMask, lf, PPS);

    IPhaser* phaser = new IPhaser;
    if(phaser->init())
	{
		phaser->addPhasingMethod( 	CXParams::getInstance()->getReconstructionParams()->algorithm.c_str(),
									CXParams::getInstance()->getReconstructionParams()->iterations);
		phaser->phase();
		phaser->writeResultsToDisk();
	}
	delete phaser;
}

static PyObject* ptychopy_algorithmnp(PyObject *args, PyObject *keywds, char *algorithm)
{
    // Default value for the algorithm parameters
    char *jobID="";
    char *fp="";
    int fs=0;
    char* hdf5path="";
    int dpf = 1;
    double beamSize=400e-9;
    char *probeGuess="";
    char *objectGuess="";
    char *lf="";
    int size=512;
    int qx=128;
    int qy=128;
    int nx=256;
    int ny=256;
    int scanDimsx=26;
    int scanDimsy=26;
    int spiralScan=0;
    int flipScanAxis=0;
    int mirror1stScanAxis=0;
    int mirror2ndScanAxis=0;
    double stepx=40e-9;
    double stepy=40e-9;
    int probeModes=1;
    double lambd=2.3843e-10;
    double dx_d=172e-6;
    double z=2.2;
    int iter=100;
    int T=0;
    int jitterRadius=0;
    double delta_p=0.1;
    int threshold=0;
    int rotate90=0;
    int sqrtData=0;
    int fftShiftData=0;
    int binaryOutput=0;
    int simulate=0;
    int phaseConstraint=1;
    int updateProbe=10;
    int updateModes=20;
    int beamstopMask=0;
    int PPS=20;

    PyObject *diffractionNP_obj = NULL;
    PyObject *positionNP_obj = NULL;
    PyObject *objectNP_obj = NULL;
    PyObject *probeNP_obj = NULL;

    static char *kwlist[] = {"jobID", "diffractionNP", "positionNP", "objectNP", "probeNP", "fp", "fs", "hdf5path", "dpf", "beamSize", "probeGuess", "objectGuess", \
    "size", "qx", "qy", "nx", "ny", "scanDimsx", "scanDimsy", "spiralScan", "flipScanAxis", "mirror1stScanAxis", \
    "mirror2ndScanAxis", "stepx", "stepy", "probeModes", "lambd", "dx_d", "z", "iter", "T", "jitterRadius", \
    "delta_p",  "threshold", "rotate90", "sqrtData", "fftShiftData", "binaryOutput", "simulate", \
    "phaseConstraint", "updateProbe", "updateModes", "beamstopMask", "lf", "PPS", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|sOOOOsisidssiiiiiiiiiiiddidddiiidiiiiiiiiiisi", kwlist,
                &jobID, &diffractionNP_obj, &positionNP_obj, &objectNP_obj, &probeNP_obj, &fp, &fs, &hdf5path, &dpf, &beamSize, &probeGuess, &objectGuess, \
                &size, &qx, &qy, &nx, &ny, &scanDimsx, &scanDimsy, &spiralScan, &flipScanAxis, &mirror1stScanAxis, \
                &mirror2ndScanAxis, &stepx, &stepy, &probeModes, &lambd, &dx_d, &z, &iter, &T, &jitterRadius, \
                &delta_p,  &threshold, &rotate90, &sqrtData, &fftShiftData, &binaryOutput, &simulate, \
                &phaseConstraint, &updateProbe, &updateModes, &beamstopMask, &lf, &PPS))

    printf("Running algorithm %s. \n", algorithm);

    double ***diffractionNP_list;
    double **positionNP_list = NULL;
    complex_t *objectNP_list = NULL;
    complex_t *probeNP_list = NULL;
    //Create C arrays from numpy objects:

    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];

    if(diffractionNP_obj!=NULL)
    {
        if(PyArray_AsCArray(&diffractionNP_obj, (void ***)&diffractionNP_list, dims, 3, descr) < 0)
        {
            PyErr_SetString(PyExc_TypeError, "error converting diffraction numpy array to c array");
        }
    }
    else
    {
        printf("Please input a diffraction pattern numpy array\n");
        Py_RETURN_NONE;
    }

    if(positionNP_obj!=NULL)
    {
        printf("Use input position numpy array\n");
        if(PyArray_AsCArray(&positionNP_obj, (void **)&positionNP_list, dims, 2, descr) < 0)
        {
            PyErr_SetString(PyExc_TypeError, "error converting position numpy array to c array");
        }
    }
    else
    {
        printf("Use default grid position\n");
    }

    if(objectNP_obj!=NULL)
    {
        PyArrayObject * yarr=NULL;
        int DTYPE = PyArray_ObjectType(objectNP_obj, NPY_FLOAT);
        int iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
        yarr = (PyArrayObject *)PyArray_FROM_OTF(objectNP_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
        if (yarr != NULL)
        {
            if (PyArray_NDIM(yarr) != 2)
            {
                Py_CLEAR(yarr);
                PyErr_SetString(PyExc_ValueError, "Expected 2 dimensional object array");
                return NULL;
            }
            npy_intp * dimsObject = PyArray_DIMS(yarr);
            npy_intp i,j;
            double * p;
            if (iscomplex)
            {
            	objectNP_list = new complex_t[scanDimsx*scanDimsy];
                for (i=0;i<dimsObject[0];i++)
                    for (j=0;j<dimsObject[1];j++)
                    {
                        p = (double*)PyArray_GETPTR2(yarr, i,j);
                        real_t real = (real_t)*p;
                        real_t imag = (real_t)(*(p+1));
                        objectNP_list[j+i*dimsObject[1]].x= real;
                        objectNP_list[j+i*dimsObject[1]].y= imag;
//                        printf("2D complex: %f + %fi\n", objectNP_list[j+i*dimsObject[1]].x, objectNP_list[j+i*dimsObject[1]].y);
                    }
            }
            Py_CLEAR(yarr);
        }
        else
        {
            Py_INCREF(Py_None);
            printf("The object array passing failed\n");
            return Py_None;
        }
    }
    else
    {
        printf("Use default object guess\n");
    }

    if(probeNP_obj!=NULL)
    {
        PyArrayObject * yarr=NULL;
        int DTYPE = PyArray_ObjectType(probeNP_obj, NPY_FLOAT);
        int iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
        yarr = (PyArrayObject *)PyArray_FROM_OTF(probeNP_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
        if (yarr != NULL)
        {
            if (PyArray_NDIM(yarr) != 2)
            {
                Py_CLEAR(yarr);
                PyErr_SetString(PyExc_ValueError, "Expected 2 dimensional probe array");
                return NULL;
            }
            npy_intp * dimsProbe = PyArray_DIMS(yarr);
            npy_intp i,j;
            double * p;
            if (iscomplex)
            {
            	probeNP_list = new complex_t[size*size];
                for (i=0;i<dimsProbe[0];i++)
                    for (j=0;j<dimsProbe[1];j++)
                    {
                        p = (double*)PyArray_GETPTR2(yarr, i,j);
                        real_t real = (real_t)*p;
                        real_t imag = (real_t)(*(p+1));
                        probeNP_list[j+i*dimsProbe[1]].x= real;
                        probeNP_list[j+i*dimsProbe[1]].y= imag;
        //                printf("2D complex: %f + i%f\n", real, imag);
                    }
            }
            Py_CLEAR(yarr);
        }
        else
        {
            Py_INCREF(Py_None);
            printf("The probe array passing failed\n");
            return Py_None;
        }
    }
    else
    {
        printf("Use default probe guess\n");
    }

//    printf("2D complex: %f + i%f, 3D: %f.\n", crealf(objectNP_list[3][1]), cimagf(objectNP_list[3][1]), diffractionNP_list[1][0][2]);
//    printf("3D: %f: .\n", diffractionNP_list[0][0][2]);
//      printf("1D: %f.\n", *((double*)(&diffractionNP_list[0][0][0])+6));
//    printf("2D: %f, 3D: %f.\n", positionNP_list[3][1], diffractionNP_list[1][0][2]);
//    printf("1D: %f, 2D: %f, 3D: %f.\n", list1[2], list2[3][1], list3[1][0][2]);

    CXParams::getInstance()->parseFromCPython(jobID, algorithm, fp, fs, hdf5path, dpf, beamSize, probeGuess, objectGuess, \
                size, qx, qy, nx, ny, scanDimsx, scanDimsy, spiralScan, flipScanAxis, mirror1stScanAxis, \
                mirror2ndScanAxis, stepx, stepy, probeModes, lambd, dx_d, z, iter, T, jitterRadius, \
                delta_p,  threshold, rotate90, sqrtData, fftShiftData, binaryOutput, simulate, \
                phaseConstraint, updateProbe, updateModes, beamstopMask, lf, PPS, diffractionNP_list, objectNP_list, probeNP_list, \
                positionNP_list);

    IPhaser* phaser = new IPhaser;
    if(phaser->init())
	{
		phaser->addPhasingMethod( 	CXParams::getInstance()->getReconstructionParams()->algorithm.c_str(),
									CXParams::getInstance()->getReconstructionParams()->iterations);
		phaser->phase();
		phaser->writeResultsToDisk();
	}
	delete phaser;

	Py_RETURN_NONE;
}

static PyObject *ptycholib_epie(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithm(args, keywds, "ePIE");

    return Py_True;
}
static PyObject *ptycholib_dm(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithm(args, keywds, "DM");

    return Py_True;
}

static PyObject *ptycholib_mls(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithm(args, keywds, "MLs");

    return Py_True;
}

static PyObject *ptycholib_epienp(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithmnp(args, keywds, "ePIE");
    return Py_True;
}

static PyObject *ptycholib_dmnp(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithmnp(args, keywds, "DM");
    return Py_True;
}

static PyObject *ptycholib_mlsnp(PyObject *self, PyObject *args, PyObject *keywds)
{

    ptychopy_algorithmnp(args, keywds, "MLs");
    return Py_True;
}

static size_t makeargs(char **ap, size_t n, char *s)
{
    int c, inarg = 0;
    size_t save;
    for (save = n--; (c = (unsigned char) *s) && n; s++)
        if (inarg && isspace(c))
        {
            *s = '\0';
            n--;
            inarg = 0;
        } else if (!inarg && !isspace(c))
        {
            *ap++ = s;
            inarg = 1;
        }
    *ap = 0;
    return save - n;
}

static size_t countargs(const char *s)
{
    int inarg = 0, c;
    size_t n = 0;


    while ((c = (unsigned char) *s++))
        if (inarg && isspace(c)) {
            inarg = 0;
        } else if (!inarg && !isspace(c)) {
            n++;
            inarg = 1;
        }
    return n;
}

static PyObject *ptycholib_epiecmdstr(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";
    static char *kwlist[] = {"cmdstr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;
    size_t len = countargs(cmdstr) + 1;
    char **r;
    if (!(r = (char **)malloc((len+1) * sizeof *r)))
        return 0;
    if (!(*r = (char *)malloc(strlen(cmdstr) + 1)))
    {
        free(r);
        return 0;
    }
    makeargs(r+1, len, strcpy(*r, cmdstr));
    CXParams::getInstance()->parseFromCommandLine(len-1, r+1);
    IPhaser* phaser = new IPhaser;

    if(phaser->init())
    {
        phaser->addPhasingMethod("ePIE", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phase();
    }

    phaser->writeResultsToDisk();
    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);
}

static PyObject *ptycholib_dmcmdstr(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";
    static char *kwlist[] = {"cmdstr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;
    size_t len = countargs(cmdstr) + 1;
    char **r;
    if (!(r = (char **)malloc((len+1) * sizeof *r)))
        return 0;
    if (!(*r = (char *)malloc(strlen(cmdstr) + 1)))
    {
        free(r);
        return 0;
    }
    makeargs(r+1, len, strcpy(*r, cmdstr));
    CXParams::getInstance()->parseFromCommandLine(len-1, r+1);
    IPhaser* phaser = new IPhaser;

    if(phaser->init())
    {
        phaser->addPhasingMethod("DM", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phase();
    }

    phaser->writeResultsToDisk();
    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);

}

static PyObject *ptycholib_mlscmdstr(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";
    static char *kwlist[] = {"cmdstr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;
    printf("the command line string is %s \n", cmdstr);
    size_t len = countargs(cmdstr) + 1;
    char **r;
    if (!(r = (char **)malloc((len+1) * sizeof *r)))
        return 0;
    if (!(*r = (char *)malloc(strlen(cmdstr) + 1)))
    {
        free(r);
        return 0;
    }
    makeargs(r+1, len, strcpy(*r, cmdstr));
    CXParams::getInstance()->parseFromCommandLine(len-1, r+1);
    IPhaser* phaser = new IPhaser;

    if(phaser->init())
    {
        phaser->addPhasingMethod("MLs", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phase();
    }

    phaser->writeResultsToDisk();
    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);
}

static PyObject* ptychopy_algorithmnpinit(PyObject *args, PyObject *keywds, char *algorithm)
{
    // Default value for the algorithm parameters
    char *jobID="";
    char *fp="";
    int fs=0;
    char* hdf5path="";
    int dpf = 1;
    double beamSize=400e-9;
    char *probeGuess="";
    char *objectGuess="";
    char *lf="";
    int size=512;
    int qx=128;
    int qy=128;
    int nx=256;
    int ny=256;
    int scanDimsx=26;
    int scanDimsy=26;
    int spiralScan=0;
    int flipScanAxis=0;
    int mirror1stScanAxis=0;
    int mirror2ndScanAxis=0;
    double stepx=40e-9;
    double stepy=40e-9;
    int probeModes=1;
    double lambd=2.3843e-10;
    double dx_d=172e-6;
    double z=2.2;
    int iter=100;
    int T=0;
    int jitterRadius=0;
    double delta_p=0.1;
    int threshold=0;
    int rotate90=0;
    int sqrtData=0;
    int fftShiftData=0;
    int binaryOutput=0;
    int simulate=0;
    int phaseConstraint=1;
    int updateProbe=10;
    int updateModes=20;
    int beamstopMask=0;
    int PPS=20;

    PyObject *diffractionNP_obj = NULL;
    PyObject *positionNP_obj = NULL;
    PyObject *objectNP_obj = NULL;
    PyObject *probeNP_obj = NULL;

    static char *kwlist[] = {"jobID", "diffractionNP", "positionNP", "objectNP", "probeNP", "fp", "fs", "hdf5path", "dpf", "beamSize", "probeGuess", "objectGuess", \
    "size", "qx", "qy", "nx", "ny", "scanDimsx", "scanDimsy", "spiralScan", "flipScanAxis", "mirror1stScanAxis", \
    "mirror2ndScanAxis", "stepx", "stepy", "probeModes", "lambd", "dx_d", "z", "iter", "T", "jitterRadius", \
    "delta_p",  "threshold", "rotate90", "sqrtData", "fftShiftData", "binaryOutput", "simulate", \
    "phaseConstraint", "updateProbe", "updateModes", "beamstopMask", "lf", "PPS", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|sOOOOsisidssiiiiiiiiiiiddidddiiidiiiiiiiiiisi", kwlist,
                &jobID, &diffractionNP_obj, &positionNP_obj, &objectNP_obj, &probeNP_obj, &fp, &fs, &hdf5path, &dpf, &beamSize, &probeGuess, &objectGuess, \
                &size, &qx, &qy, &nx, &ny, &scanDimsx, &scanDimsy, &spiralScan, &flipScanAxis, &mirror1stScanAxis, \
                &mirror2ndScanAxis, &stepx, &stepy, &probeModes, &lambd, &dx_d, &z, &iter, &T, &jitterRadius, \
                &delta_p,  &threshold, &rotate90, &sqrtData, &fftShiftData, &binaryOutput, &simulate, \
                &phaseConstraint, &updateProbe, &updateModes, &beamstopMask, &lf, &PPS))

    printf("Running algorithm %s. \n", algorithm);

    double ***diffractionNP_list;
    double **positionNP_list = NULL;
    complex_t *objectNP_list = NULL;
    complex_t *probeNP_list = NULL;
    //Create C arrays from numpy objects:

    int typenum = NPY_DOUBLE;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(typenum);
    npy_intp dims[3];

    if(diffractionNP_obj!=NULL)
    {
        if(PyArray_AsCArray(&diffractionNP_obj, (void ***)&diffractionNP_list, dims, 3, descr) < 0)
        {
            PyErr_SetString(PyExc_TypeError, "error converting diffraction numpy array to c array");
        }
    }
    else
    {
        printf("Please input a diffraction pattern numpy array\n");
        Py_RETURN_NONE;
    }

    if(positionNP_obj!=NULL)
    {
        printf("Use input position numpy array\n");
        if(PyArray_AsCArray(&positionNP_obj, (void **)&positionNP_list, dims, 2, descr) < 0)
        {
            PyErr_SetString(PyExc_TypeError, "error converting position numpy array to c array");
        }
    }
    else
    {
        printf("Use default grid position\n");
    }

    if(objectNP_obj!=NULL)
    {
        PyArrayObject * yarr=NULL;
        int DTYPE = PyArray_ObjectType(objectNP_obj, NPY_FLOAT);
        int iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
        yarr = (PyArrayObject *)PyArray_FROM_OTF(objectNP_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
        if (yarr != NULL)
        {
            if (PyArray_NDIM(yarr) != 2)
            {
                Py_CLEAR(yarr);
                PyErr_SetString(PyExc_ValueError, "Expected 2 dimensional object array");
                return NULL;
            }
            npy_intp * dimsObject = PyArray_DIMS(yarr);
            npy_intp i,j;
            double * p;
            if (iscomplex)
            {
            	objectNP_list = new complex_t[scanDimsx*scanDimsy];
                for (i=0;i<dimsObject[0];i++)
                    for (j=0;j<dimsObject[1];j++)
                    {
                        p = (double*)PyArray_GETPTR2(yarr, i,j);
                        real_t real = (real_t)*p;
                        real_t imag = (real_t)(*(p+1));
                        objectNP_list[j+i*dimsObject[1]].x= real;
                        objectNP_list[j+i*dimsObject[1]].y= imag;
//                        printf("2D complex: %f + %fi\n", objectNP_list[j+i*dimsObject[1]].x, objectNP_list[j+i*dimsObject[1]].y);
                    }
            }
            Py_CLEAR(yarr);
        }
        else
        {
            Py_INCREF(Py_None);
            printf("The object array passing failed\n");
            return Py_None;
        }
    }
    else
    {
        printf("Use default object guess\n");
    }

    if(probeNP_obj!=NULL)
    {
        PyArrayObject * yarr=NULL;
        int DTYPE = PyArray_ObjectType(probeNP_obj, NPY_FLOAT);
        int iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);
        yarr = (PyArrayObject *)PyArray_FROM_OTF(probeNP_obj, DTYPE, NPY_ARRAY_IN_ARRAY);
        if (yarr != NULL)
        {
            if (PyArray_NDIM(yarr) != 2)
            {
                Py_CLEAR(yarr);
                PyErr_SetString(PyExc_ValueError, "Expected 2 dimensional probe array");
                return NULL;
            }
            npy_intp * dimsProbe = PyArray_DIMS(yarr);
            npy_intp i,j;
            double * p;
            if (iscomplex)
            {
            	probeNP_list = new complex_t[size*size];
                for (i=0;i<dimsProbe[0];i++)
                    for (j=0;j<dimsProbe[1];j++)
                    {
                        p = (double*)PyArray_GETPTR2(yarr, i,j);
                        real_t real = (real_t)*p;
                        real_t imag = (real_t)(*(p+1));
                        probeNP_list[j+i*dimsProbe[1]].x= real;
                        probeNP_list[j+i*dimsProbe[1]].y= imag;
        //                printf("2D complex: %f + i%f\n", real, imag);
                    }
            }
            Py_CLEAR(yarr);
        }
        else
        {
            Py_INCREF(Py_None);
            printf("The probe array passing failed\n");
            return Py_None;
        }
    }
    else
    {
        printf("Use default probe guess\n");
    }

    CXParams::getInstance()->parseFromCPython(jobID, algorithm, fp, fs, hdf5path, dpf, beamSize, probeGuess, objectGuess, \
                size, qx, qy, nx, ny, scanDimsx, scanDimsy, spiralScan, flipScanAxis, mirror1stScanAxis, \
                mirror2ndScanAxis, stepx, stepy, probeModes, lambd, dx_d, z, iter, T, jitterRadius, \
                delta_p,  threshold, rotate90, sqrtData, fftShiftData, binaryOutput, simulate, \
                phaseConstraint, updateProbe, updateModes, beamstopMask, lf, PPS, diffractionNP_list, objectNP_list, probeNP_list, \
                positionNP_list);


    phaser = new IPhaser;
    if(phaser->init())
	{
		phaser->addPhasingMethod( 	CXParams::getInstance()->getReconstructionParams()->algorithm.c_str(),
									CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phaseinit();
	}

	Py_RETURN_NONE;
}

static PyObject *ptycholib_epieinit(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";
    phaser = new IPhaser;
    istep = 0;
    static char *kwlist[] = {"cmdstr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;

    size_t len = countargs(cmdstr) + 1;
    char **r;

    if (!(r = (char **)malloc((len+1) * sizeof *r)))
        return 0;

    if (!(*r = (char *)malloc(strlen(cmdstr) + 1)))
    {
        free(r);
        return 0;
    }

    makeargs(r+1, len, strcpy(*r, cmdstr));
    CXParams::getInstance()->parseFromCommandLine(len-1, r+1);
    if(phaser->init())
    {
        phaser->addPhasingMethod("ePIE", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phaseinit();
    }

    free(*r);
    free(r);
    return Py_True;

}

static PyObject *ptycholib_epienpinit(PyObject *self, PyObject *args, PyObject *keywds)
{
    ptychopy_algorithmnpinit(args, keywds, "ePIE");
    return Py_True;
}

static PyObject *ptycholib_epiepost(PyObject *self, PyObject *args, PyObject *keywds)
{

    phaser->phasepost();
    phaser->writeResultsToDisk();
    if(phaser != NULL)
        delete phaser;
    return Py_True;

}

static PyObject *ptycholib_epiestep(PyObject *self, PyObject *args, PyObject *keywds)
{

    istep++;
    phaser->phasestepvis(istep);
    return Py_True;

}

static PyObject *ptycholib_epieresobj(PyObject *self, PyObject *args, PyObject *keywds)
{

    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);

}

static PyObject *ptycholib_epieresprobe(PyObject *self, PyObject *args, PyObject *keywds)
{

    complex_t* h_probeArray = phaser->getProbe()->getModes()->getPtr()->getHostPtr<complex_t>();
    int x = phaser->getProbe()->getModes()->getPtr()->getX();
    int y = phaser->getProbe()->getModes()->getPtr()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_probeArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);

}

static PyObject *ptycholib_mlsinit(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";
    phaser = new IPhaser;
    istep = 0;
    static char *kwlist[] = {"cmdstr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;

    size_t len = countargs(cmdstr) + 1;
    char **r;

    if (!(r = (char **)malloc((len+1) * sizeof *r)))
        return 0;

    if (!(*r = (char *)malloc(strlen(cmdstr) + 1)))
    {
        free(r);
        return 0;
    }

    makeargs(r+1, len, strcpy(*r, cmdstr));
    CXParams::getInstance()->parseFromCommandLine(len-1, r+1);
    if(phaser->init())
    {
        phaser->addPhasingMethod("MLs", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phaseinit();
    }

    free(*r);
    free(r);
    return Py_True;

}

static PyObject *ptycholib_mlspost(PyObject *self, PyObject *args, PyObject *keywds)
{

    phaser->phasepost();
    phaser->writeResultsToDisk();
    if(phaser != NULL)
        delete phaser;
    return Py_True;

}

static PyObject *ptycholib_mlsstep(PyObject *self, PyObject *args, PyObject *keywds)
{

    istep++;
    phaser->phasestepvis(istep);
    return Py_True;

}

static PyObject *ptycholib_mlsresobj(PyObject *self, PyObject *args, PyObject *keywds)
{

    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);

}

static PyObject *ptycholib_mlsresprobe(PyObject *self, PyObject *args, PyObject *keywds)
{

    complex_t* h_probeArray = phaser->getProbe()->getModes()->getPtr()->getHostPtr<complex_t>();
    int x = phaser->getProbe()->getModes()->getPtr()->getX();
    int y = phaser->getProbe()->getModes()->getPtr()->getY();

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_probeArray);
    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);

}
