#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>

#ifdef __INTEL_COMPILER
#include <mkl_cblas.h>
#else
#include <cblas.h>
#define kmp_set_blocktime(k) 
#endif

static inline float minf (float a, float b) { return a < b ? a : b; }
static inline float maxf (float a, float b) { return a < b ? b : a; }


PyObject * compute_overlap (float bbx1, float bby1, float bbx2, float bby2,
                            int fdimy, int fdimx, int dimy, int dimx,
                            float scale, int pady, int padx, int h, int w) {
    int im_area = h*w;
    int bbox_area = (bbx2-bbx1)*(bby2-bby1);
    int im_clip = ((double)bbox_area / (double)im_area) < 0.7;
    npy_intp dims[2] = { dimy, dimx };
    int x, y;

    PyArrayObject * pyoverlap = (PyArrayObject*)PyArray_SimpleNew ((npy_intp)2, dims, NPY_FLOAT);

    for (x = 0; x < dimx; ++x) {
        for (y = 0; y < dimy; ++y) {
            float x1 = (x - padx) * scale;
            float y1 = (y - pady) * scale;
            float x2 = x1 + fdimx*scale - 1;
            float y2 = y1 + fdimy*scale - 1;

            if (im_clip) {
                x1 = minf(maxf(x1, 0.0f), w-1);
                y1 = minf(maxf(y1, 0.0f), h-1);
                x2 = minf(maxf(x2, 0.0f), w-1);
                y2 = minf(maxf(y2, 0.0f), h-1);
            }

            float xx1 = maxf(x1, bbx1);
            float yy1 = maxf(y1, bby1);
            float xx2 = minf(x2, bbx2);
            float yy2 = minf(y2, bby2);

            float intw = xx2 - xx1 + 1;
            float inth = yy2 - yy1 + 1;

            if (intw > 0 && inth > 0) {
                float filterw = x2 - x1 + 1;
                float filterh = y2 - y1 + 1;
                float filter_area = filterw*filterh;
                float int_area = intw*inth;
                float union_area = filter_area + bbox_area - int_area;

                *(float*)PyArray_GETPTR2(pyoverlap, y, x) = int_area / union_area;
            } else {
                *(float*)PyArray_GETPTR2(pyoverlap, y, x) = 0;
            }
        }
    }

    return PyArray_Return (pyoverlap);
}

PyObject * objective_function (PyObject * pyexamples) {
    int i;
    int j;
    int k;

    for (i = 0; i < PyList_Size(pyexamples); ++i) {
        PyObject * pyexample = PyList_GetItem (pyexamples, i);
        if (!PyList_Check(pyexample)) {
            PyErr_SetString(PyExc_TypeError, "example list elements must be list.");
            return NULL;
        }

        for (j = 0; j < PyList_Size(pyexample); ++j) {

            PyObject * pyentry = PyList_GetItem(pyexample, j);
            if (!PyTuple_Check(pyentry)) {
                PyErr_SetString(PyExc_TypeError, "entries must be tuples.");
                return NULL;
            }

            if (PyTuple_Size(pyentry) != 2) {
                PyErr_SetString(PyExc_TypeError, "entry tuples must be of size 2");
                return NULL;
            }

            PyObject * pyfeatures = PyTuple_GetItem(pyentry, 0);
            if (!PyDict_Check(pyfeatures)) {
                PyErr_SetString(PyExc_TypeError, "features must be a dict");
                return NULL;
            }

            PyObject * pyfeatureslist = PyDict_Items(pyfeatures);

            float score = 0;
            for (k = 0; k < PyDict_Size(pyfeatureslist); ++k) {
                PyObject * pyfeature = PyList_GetItem(pyfeatureslist, k);
                PyObject * pyblock = PyTuple_GetItem(pyfeature, 0);

                PyObject * pyw = PyTuple_GetItem(pyfeature, 1);
                if (!PyArray_Check(pyw)) {
                    PyErr_SetString(PyExc_TypeError, "w must be a numpy array");
                    return NULL;
                }

                PyArrayObject * pywarray = (PyArrayObject*)pyw;

                
            }

            PyObject * pydetection = PyTuple_GetItem(pyentry, 1);
        }
    }

    return Py_BuildValue("i", 0);
}

PyObject * gradient (PyObject * examples, PyObject * gradients) {
    Py_RETURN_NONE;
}

static PyObject * ComputeOverlap(PyObject * self, PyObject * args)
{
    float x1, x2, y1, y2;
    int fdimy, fdimx, dimy, dimx;
    float scale;
    int pady, padx;
    int w, h;
    if (!PyArg_ParseTuple(args, "ffffiiiifiiii", &x1, &y1, &x2, &y2, &fdimy, &fdimx, &dimy, &dimx, &scale, &pady, &padx, &h, &w)) 
        return NULL;
    return compute_overlap(x1, y1, x2, y2, fdimy, fdimx, dimy, dimx, scale, pady, padx, h, w);
}

static PyObject * ObjectiveFunction (PyObject * self, PyObject * args) {
    PyObject * pyexamples;
    if (!PyArg_ParseTuple (args, "O!", &PyList_Type, &pyexamples)) {
        return NULL;
    }

    return objective_function (pyexamples);
}

static PyObject * Gradient (PyObject * self, PyObject * args) {
    PyObject * pyexamples;
    PyObject * pygradients;
    if (!PyArg_ParseTuple (args, "O!O!", &PyList_Type, &pyexamples, &PyDict_Type, &pygradients)) {
        return NULL;
    }

    return gradient (pyexamples, pygradients);
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_train",
    "Routines for training the DPM.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef _train_methods[] = {
    {"ComputeOverlap", ComputeOverlap, METH_VARARGS, "Compute detection overlaps with bbox."},
    {"ObjectiveFunction", ObjectiveFunction, METH_VARARGS, "WL-SSVM objective function."},
    {"Gradient", Gradient, METH_VARARGS, "WL-SSVM gradient."},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__train(void)
#else
PyMODINIT_FUNC
init_train(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("_train", _train_methods, "Compute detection overlaps with bbox.");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
