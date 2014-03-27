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

static inline int min (int a, int b) { return a < b ? a : b; }
static inline int max (int a, int b) { return a < b ? b : a; }

static inline int square(int x) { return x*x; }

static void max_filter_1d(const float *vals, float *out_vals, int32_t *I, 
                          int s, int step, int n, float a, float b) {
  int i;
  for (i = 0; i < n; i++) {
    float max_val = -INFINITY;
    int argmax     = 0;
    int first      = max(0, i-s);
    int last       = min(n-1, i+s);
    int j;
    for (j = first; j <= last; j++) {
      float val = *(vals + j*step) - a*square(i-j) - b*(i-j);
      if (val > max_val) {
        max_val = val;
        argmax  = j;
      }
    }
    *(out_vals + i*step) = max_val;
    *(I + i*step) = argmax;
  }
}

PyObject * deformation_cost (PyArrayObject * pydata, float ax, float bx, float ay, float by, int s) {
  npy_intp * dims = PyArray_DIMS(pydata);
  npy_intp * stride = PyArray_STRIDES(pydata);

  if (PyArray_NDIM(pydata) != 2) {
    PyErr_SetString(PyExc_TypeError, "data must be 2 dimensional.");
    return NULL;
  }

  if (PyArray_DESCR(pydata)->type_num != NPY_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "data must be single precision floating point.");
    return NULL;
  }

  if (stride[0] != dims[1]*sizeof(float)) {
    PyErr_SetString(PyExc_TypeError, "Stride[0] must be sizeof(float).");
    return NULL;
  }

  if (stride[1] != sizeof(float)) {
    PyErr_SetString(PyExc_TypeError, "Stride[1] must be Dims[0]*sizeof(float).");
    return NULL;
  }
  
  PyArrayObject * pydeformed = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, dims, NPY_FLOAT);
  PyArrayObject * pyIx = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, dims, NPY_INT32);
  PyArrayObject * pyIy = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, dims, NPY_INT32);

  float *tmpM = (float *)calloc(dims[0]*dims[1], sizeof(float));
  int32_t *tmpIx = (int32_t*)calloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t*)calloc(dims[0]*dims[1], sizeof(int32_t));

  int x, y;

  for (y = 0; y < dims[0]; y++)
    max_filter_1d((float*)PyArray_GETPTR2(pydata, y, 0), tmpM+y*dims[1], tmpIx+y*dims[1], s, 1, dims[1], ax, bx);

  for (x = 0; x < dims[1]; x++)
    max_filter_1d(tmpM+x, (float*)PyArray_GETPTR2(pydeformed, 0, x), tmpIy+x, s, dims[1], dims[0], ay, by);

  for (x = 0; x < dims[1]; ++x) {
    for (y = 0; y < dims[0]; ++y) {
      *(int32_t*)PyArray_GETPTR2(pyIy, y, x) = tmpIy[y*dims[1]+x];
      *(int32_t*)PyArray_GETPTR2(pyIx, y, x) = tmpIx[tmpIy[y*dims[1]+x]*dims[1]+x];
    }
  }

  free(tmpM);
  free(tmpIx);
  free(tmpIy);

  return Py_BuildValue("NNN", pydeformed, pyIx, pyIy);
}

PyObject * filter_image (PyArrayObject * pyfeatures, PyArrayObject * pyfilter, float bias, int width, int height) {
    npy_intp * features_dims = PyArray_DIMS(pyfeatures);
    npy_intp * filter_dims = PyArray_DIMS(pyfilter);
    int a, b, l;
    PyArrayObject * pyfiltered = NULL;
    npy_intp * features_stride = PyArray_STRIDES(pyfeatures);
    npy_intp * filtered_stride = NULL;
    npy_intp filtered_dims[2] = {0, 0};
    int tight_width;
    int tight_height;

    if (PyArray_NDIM(pyfeatures) != 3) {
        PyErr_SetString(PyExc_TypeError, "Features must be 3 dimensional.");
        return NULL;
    }

    if (PyArray_NDIM(pyfilter) != 3) {
        PyErr_SetString(PyExc_TypeError, "Filter must be 3 dimensional.");
        return NULL;
    }

    if (PyArray_DESCR(pyfeatures)->type_num != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "Features must be single precision floating point.");
        return NULL;
    }

    if (PyArray_DESCR(pyfilter)->type_num != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "Filter must be a single precision floating point.");
        return NULL;
    }

    if (features_dims[2] != 32) {
        PyErr_SetString(PyExc_TypeError, "features' feature dimsionality should be 32.");
        return NULL;
    }

    if (filter_dims[2] != 32) {
        PyErr_SetString(PyExc_TypeError, "filters' feature dimensionality should be 32.");
        return NULL;
    }

    tight_height = features_dims[0]-filter_dims[0]+1;
    tight_width = features_dims[1]-filter_dims[1]+1;

    filtered_dims[0] = height ? height : tight_height;
    filtered_dims[1] = width ? width : tight_width;

    if (filtered_dims[0] < 1 || filtered_dims[1] < 1) {
        PyErr_SetString(PyExc_TypeError, "Input features are too small for filter.");
        return NULL;
    }

    #pragma omp critical
    pyfiltered = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, filtered_dims, NPY_FLOAT);

    filtered_stride = PyArray_STRIDES(pyfiltered);

    /* zero out array */
    for (a = 0; a < tight_height; ++a) {
        for (b = 0; b < tight_width; ++b) {
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -bias;
        }
    }

    /* iterate over filter which should be tiny compared to the image */
    int i;
    int stride_src = features_stride[1]/sizeof(float);
    int stride_dst = filtered_stride[1]/sizeof(float);
    for (i = 0; i < filter_dims[0]; ++i) {
        int j;
        for (j = 0; j < filter_dims[1]; ++j) {
            int k;
            for (k = 0; k < tight_height; ++k) {
                float * out = (float*)PyArray_GETPTR2(pyfiltered, k, 0);
                /* for each layer */
                for (l = 0; l < 32; ++l) {
                    float weight = *(float*)PyArray_GETPTR3(pyfilter, i, j, l);
                    float * in = (float*)PyArray_GETPTR3(pyfeatures, i+k, j, l);
                    cblas_saxpy(tight_width, weight, in, stride_src, out, stride_dst);
                }
            }
        }
    }

    for (a = tight_height; a < filtered_dims[0]; ++a) {
        for (b = 0; b < filtered_dims[1]; ++b) {
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;
        }
    }

    for (a = 0; a < tight_height; ++a) {
        for (b = tight_width; b < filtered_dims[1]; ++b) {
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;
        }
    }

    return Py_BuildValue("N", pyfiltered);
}

static PyObject * DeformationCost(PyObject * self, PyObject * args)
{
    PyArrayObject * pydata;
    float ax = 0.0f, bx = 0.0f, ay = 0.0f, by = 0.0f;
    int s = 0;
    if (!PyArg_ParseTuple(args, "O!ffffi", &PyArray_Type, &pydata, &ax, &bx, &ay, &by, &s)) 
        return NULL;
    return deformation_cost(pydata, ax, bx, ay, by, s);
}

static PyObject * FilterImage(PyObject * self, PyObject * args)
{
    PyArrayObject * pyfeatures;
    PyArrayObject * pyfilter;
    float bias = 0.0f;
    int width = 0;
    int height = 0;
    if (!PyArg_ParseTuple(args, "O!O!|fii", &PyArray_Type, &pyfeatures, &PyArray_Type, &pyfilter, &bias, &width, &height)) 
        return NULL;
    return filter_image(pyfeatures, pyfilter, bias, width, height);
}

static PyObject * FilterImages(PyObject * self, PyObject * args)
{
    PyObject * pyfeatures_list;
    PyObject * pydims_list = NULL;
    PyArrayObject * pyfilter;
    float bias = 0.0f;
    int numfilters;
    int numdims = 0;
    int i;
    PyObject ** objs = NULL;
    PyObject ** results = NULL;
    PyObject * pyresults_list;
    if (!PyArg_ParseTuple(args, "O!O!|fO!", &PyList_Type, &pyfeatures_list, &PyArray_Type, &pyfilter, &bias, &PyList_Type, &pydims_list)) 
        return NULL;

    numfilters = PyList_Size(pyfeatures_list);
    
    if (pydims_list) {
        numdims = PyList_Size(pydims_list);
    }

    if (numdims && numdims != numfilters) {
        PyErr_SetString(PyExc_TypeError, "If pad dims are specified, then it must be the same length as the features list.");
        return NULL;
    }

    objs = (PyObject**)calloc(numfilters, sizeof(PyObject*));
    int* widths = (int*)calloc(numfilters, sizeof(int));
    int* heights = (int*)calloc(numfilters, sizeof(int));
    results = (PyObject**)calloc(numfilters, sizeof(PyObject*));

    for (i = 0; i < numfilters; ++i) {
        objs[i] = PyList_GetItem(pyfeatures_list, i);
        if (!PyArray_Check(objs[i])) {
            free(objs);
            free(widths);
            free(heights);
            free(results);
            PyErr_SetString(PyExc_TypeError, "Must contain a list of numpy arrays.");
            return NULL;
        }

        if (pydims_list) {
            PyObject * dims = PyList_GetItem(pydims_list, i);
            if (!PyTuple_Check(dims) || 2 != PyTuple_Size(dims)) {
                free(objs);
                free(widths);
                free(heights);
                free(results);
                PyErr_SetString(PyExc_TypeError, "Must contain a list of tuples.");
                return NULL;
            }

            heights[i] = PyInt_AsLong(PyTuple_GetItem(dims, 0));
            widths[i] = PyInt_AsLong(PyTuple_GetItem(dims, 1));
        } else {
            widths[i] = 0;
            heights[i] = 0;
        }
    }

    kmp_set_blocktime(0);
    #pragma omp parallel for schedule(dynamic) 
    for (i = 0; i < numfilters; ++i) { 
        PyArrayObject * pyfeatures = (PyArrayObject*)objs[i];
        results[i] = filter_image(pyfeatures, pyfilter, bias, widths[i], heights[i]);
    }

    free(objs);
    free(widths);
    free(heights);

    pyresults_list = PyList_New(numfilters);

    for (i = 0; i < numfilters; ++i) {
        PyList_SetItem(pyresults_list, i, results[i]);
    }

    free(results);

    return Py_BuildValue("N", pyresults_list);    
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_detection",
    "Native convolution detection routine.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef _detection_methods[] = {
    {"FilterImage", FilterImage, METH_VARARGS, "Compute a 2D cross correlation between a filter and image features.  Optionally add bias term."},
    {"FilterImages", FilterImages, METH_VARARGS, "Compute a 2D cross correlation between a filter and several image features in parallel.  Optionally add bias term."},
    {"DeformationCost", DeformationCost, METH_VARARGS, "Compute a fast bounded distance transform for the deformation cost."},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__detection(void)
#else
PyMODINIT_FUNC
init_detection(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("_detection", _detection_methods, "Native convolution detection routine.");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
