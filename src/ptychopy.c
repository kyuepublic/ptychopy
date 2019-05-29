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

static char epie_docstring[] = "This function performs iterative phase retrieval using the ePIE algorithm.";

static char epiecmdstr_docstring[] = "This function performs iterative phase retrieval using the ePIE algorithm.";

static char epieinit_docstring[] = "This function performs the init.";

static char epiepost_docstring[] = "This function performs the post.";

static char epiestep_docstring[] = "This function performs the step.";

static char epieresobj_docstring[] = "This function returns the object of the reconstruction.";

static char epieresprobe_docstring[] = "This function returns the probe of the reconstruction.";

/* Define function headers */
static PyObject *ptycholib_helloworld(PyObject *self, PyObject *args);
static PyObject *ptycholib_epie(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epiecmdstr(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *ptycholib_epieinit(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epiepost(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epiestep(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epieresobj(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *ptycholib_epieresprobe(PyObject *self, PyObject *args, PyObject *keywds);

static IPhaser* phaser;

static unsigned int istep = 0;

/* Define module methods */
static PyMethodDef module_methods[] = {
    {"helloworld", (PyCFunction)ptycholib_helloworld, METH_VARARGS, helloworld_docstring},
    {"epie", (PyCFunction)ptycholib_epie, METH_VARARGS|METH_KEYWORDS, epie_docstring},
    {"epiecmdstr", (PyCFunction)ptycholib_epiecmdstr, METH_VARARGS|METH_KEYWORDS, epiecmdstr_docstring},
    {"epieinit", (PyCFunction)ptycholib_epieinit, METH_VARARGS|METH_KEYWORDS, epieinit_docstring},
    {"epiepost", (PyCFunction)ptycholib_epiepost, METH_VARARGS|METH_KEYWORDS, epiepost_docstring},
    {"epiestep", (PyCFunction)ptycholib_epiestep, METH_VARARGS|METH_KEYWORDS, epiestep_docstring},
    {"epieresobj", (PyCFunction)ptycholib_epieresobj, METH_VARARGS|METH_KEYWORDS, epieresobj_docstring},
    {"epieresprobe", (PyCFunction)ptycholib_epieresprobe, METH_VARARGS|METH_KEYWORDS, epieresprobe_docstring},
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
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    printf("Hello, World!");

    Py_RETURN_NONE;
}

static PyObject *ptycholib_epie(PyObject *self, PyObject *args, PyObject *keywds)
{
    /* io */
    int io_threads=100;
    char *filename_pattern = "/data/id2data/scan060/scan060_#06d.h5";
    int start_file_number = 1501;

    /* preprocessing */
    int size = 512;
    int threshhold = 0;
    int rotate_90 = 0;
    int sqrt_data = 0;
    int fftshift_data = 0;
    int flip_scan_axis = 0;

    /* experiment */
    float beam_size=400e-9;
    float energy=5.0;
    float lambda=1.24e-9/energy;
    float dx=172e-6;
    float z=1.0;
    int simulate=1;

    /* cartesian scan */
    int scan_dims_0=55;
    int scan_dims_1=75;
    float step_0=40e-9;
    float step_1=40e-9;

    /* probe guess */
    char *probeGuess = "";

    /* phase retrieval */
    int object_array=2048;
    int i=10;
    char *job_id="gentest101";
    int qx0 = 243;
    int qy0 = 153;
    int probe_modes = 1;
    int position_jitter_radius=0;
    int share_frequency = 0;

    int dpf = 1;

    static char *kwlist[] = {"io_threads", "filename_pattern", "start_file_number", "size", "threshhold", "rotate_90", \
    "sqrt_data", "fftshift_data", "flip_scan_axis", "beam_size", "energy", "dx", "z", "simulate", "scan_dims", "step", \
    "probeGuess", "object_array", "i", "job_id", "qy0qx0", "probe_modes", "position_jitter_radius", \
    "share_frequency", "dpf", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|isiiiiiiiffffi(ii)(ff)siis(ii)iiii", kwlist,
                &io_threads, &filename_pattern, &start_file_number, &size, &threshhold, &rotate_90, &sqrt_data, \
                &fftshift_data, &flip_scan_axis, &beam_size, &energy, &dx, &z, &simulate, &scan_dims_0, \
                &scan_dims_1, &step_0, &step_1, &probeGuess, &object_array, &i, &job_id, &qy0, &qx0, \
                &probe_modes, &position_jitter_radius, &share_frequency, &dpf))
        return NULL;

    printf("Running ePIE.\n");
    printf("Io Threads: %d\n", io_threads);
    printf("Filename Pattern: %s\n", filename_pattern);
    printf("Start Filenumber: %d\n", start_file_number);
    printf("Size: %d\n", size);
    printf("Threshhold: %d\n", threshhold);
    printf("Rotate 90: %d\n", rotate_90);
    printf("Square Root Data: %d\n", sqrt_data);
    printf("FFT Shift Data: %d\n", fftshift_data);
    printf("Flip Scan Axis: %d\n", flip_scan_axis);
    printf("Beam Size: %f [nm]\n", 1e9*beam_size);
    printf("Energy: %f\n", energy);
    printf("dx: %f\n", dx);
    printf("z: %f\n", z);
    printf("Simulate: %d\n", simulate);
    printf("Scan Dims: %d, %d\n", scan_dims_0,scan_dims_1);
    printf("Step: %f, %f\n", step_0,step_1);
    printf("List Scan Filename: %s\n", probeGuess);
    printf("Object Array: %d\n", object_array);
    printf("Iterations: %d\n", i);
    printf("Job ID: %s\n", job_id);
    printf("Qx0Qy0: %d, %d\n", qx0,qy0);
    printf("Probe Modes: %d\n", probe_modes);
    printf("Position Jitter Radius: %d\n", position_jitter_radius);
    printf("Share Frequency: %d\n", share_frequency);
    printf("dpf: %d\n", dpf);

    lambda = 1.24e-9/energy;

    char str[1000];
    sprintf(str, "./ePIE -i=%d -jobID=%s -beamSize=%e -scanDims=%d,%d -step=%e,%e -size=%d \
    -objectArray=%d -lambda=%e -dx_d=%e -z=%f -simulate=%d -blind=0 -fp=%s -fs=%d \
    -flipScanAxis=%d -probeModes=%d -jitterRadius=%d -threshold=%d -rotate90=%d \
    -sqrtData=%d -fftShiftData=%d -shareFrequency=%d -dpf=%d -probeGuess=%s -qxy=%d,%d", i,
     job_id, beam_size, scan_dims_0, scan_dims_1, step_0, step_1, size,
     object_array, lambda, dx, z, simulate, filename_pattern, start_file_number, flip_scan_axis,
     probe_modes, position_jitter_radius, threshhold, rotate_90, sqrt_data, fftshift_data, share_frequency, dpf, probeGuess, qy0, qx0);

    printf("the string is %s \n", str);

//    sprintf(str, "./ePIE -jobID=sim512 -beamSize=110e-9 -scanDims=30,30 -step=50e-9,50e-9 -i=200 -size=512 -lambda=2.4796837508399954e-10 -dx_d=172e-6 -z=1 -simulate=1 -blind=0");

//    sprintf(str, "/local/kyue/anlproject/ptography/ptycholib/src/ePIE -jobID=gentest100 -fp=/data/id2data/scan060/scan060_#06d.h5 -fs=1501 -beamSize=400e-9 -qxy=153,243 -scanDims=55,75 -step=200e-9,200e-9 -probeModes=3 -rotate90=0 -sqrtData -fftShiftData -threshold=1 -size=256 -objectArray=1600 -lambda=1.21e-10 -dx_d=172e-6 -z=4.2 -i=50 -blind=0");



    char *agv[25];//The number is the total number of the parameters
    char *tokens = strtok(str, " ");
    int j=0;
    while (tokens != NULL) {
//        printf ("%s\n", tokens);
        agv[j++] = tokens;
        tokens = strtok(NULL, " ");
    }

//    printf("error here \n");
    CXParams::getInstance()->parseFromCommandLine(25, agv);

    IPhaser* phaser = new IPhaser;

//    IPhaser* phaser = new SimplePhaser;

    if(phaser->init())
    {
        phaser->addPhasingMethod("ePIE", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phase();
    }

    phaser->writeResultsToDisk();

//    printf("error here \n");
//    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();
//    delete phaser;
//
//    npy_intp dims[2] = {2048,2048};
//    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);

//    PyArrayObject *recon_ob = NULL;
//
//    if (recon_ob==NULL)
//    {
//        return NULL;
//    }
//
//    Py_INCREF(recon_ob);
//    return PyArray_Return(recon_ob);
    return Py_False;
}

size_t makeargs(char **ap, size_t n, char *s)
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

size_t countargs(const char *s)
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
    /* io */
    int io_threads=100;
    char *filename_pattern = "";
    int start_file_number = 1501;

    /* preprocessing */
    int size = 512;
    int threshhold = 0;
    int rotate_90 = 0;
    int sqrt_data = 0;
    int fftshift_data = 0;
    int flip_scan_axis = 0;

    /* experiment */
    float beam_size=400e-9;
    float energy=5.0;
    float lambda=1.24e-9/energy;
    float dx=172e-6;
    float z=1.0;
    int simulate=1;

    /* cartesian scan */
    int scan_dims_0=55;
    int scan_dims_1=75;
    float step_0=40e-9;
    float step_1=40e-9;

    /* probe guess */
    char *probeGuess = "";

    /* phase retrieval */
    int object_array=2048;
    int i=10;
    char *job_id="gentest101";
    int qx0 = 243;
    int qy0 = 153;
    int probe_modes = 1;
    int position_jitter_radius=0;
    int share_frequency = 0;

    int dpf = 1;

    char *cmdstr = "";
//    char cmdstr[1000];

    static char *kwlist[] = {"cmdstr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;


    lambda = 1.24e-9/energy;

    char str[1000];


    printf("the command line string is %s \n", cmdstr);

// Make the command line parameters
    size_t len = countargs(cmdstr) + 1;
    char **r;

    if (!(r = (char **)malloc(len * sizeof *r)))
        return 0;

    makeargs(r, len, cmdstr);

    CXParams::getInstance()->parseFromCommandLine(len-1, r);

    IPhaser* phaser = new IPhaser;

    if(phaser->init())
    {
        phaser->addPhasingMethod("ePIE", CXParams::getInstance()->getReconstructionParams()->iterations);
        phaser->phase();
//        phaser->phaseVisStep();
    }

    phaser->writeResultsToDisk();

    complex_t* h_objectArray = phaser->getSample()->getObjectArray()->getHostPtr<complex_t>();

    int x = phaser->getSample()->getObjectArray()->getX();
    int y = phaser->getSample()->getObjectArray()->getY();

//    delete phaser;

    npy_intp dims[2] = {x,y};
    PyArrayObject *recon_ob = (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, h_objectArray);


    Py_INCREF(recon_ob);
    return PyArray_Return(recon_ob);
//    return Py_False;
}

static PyObject *ptycholib_epieinit(PyObject *self, PyObject *args, PyObject *keywds)
{

    char *cmdstr = "";

//    if(phaser != NULL)
//        delete phaser;
//
    phaser = new IPhaser;

    istep = 0;

    static char *kwlist[] = {"cmdstr", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", kwlist,&cmdstr))
        return NULL;


    char str[1000];
//    printf("the command line string is %s \n", cmdstr);

// Make the command line parameters
//    size_t len = countargs(cmdstr) + 1;
//    char **r;
//
//    if (!(r = (char **)malloc(len * sizeof *r)))
//        return 0;
//
//    makeargs(r, len, cmdstr);

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
