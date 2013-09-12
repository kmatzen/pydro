#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>

#ifdef __USE_MKL__
#include <mkl_cblas.h>
#else
#include <cblas.h>
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

  float *tmpM = (float *)calloc(dims[0]*dims[1], sizeof(float));
  int32_t *tmpIx = (int32_t *)calloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)calloc(dims[0]*dims[1], sizeof(int32_t));

  int x, y;

  for (y = 0; y < dims[0]; y++)
    max_filter_1d(PyArray_GETPTR2(pydata, y, 0), tmpM+y*dims[1], tmpIx+y*dims[1], s, 1, dims[1], ax, bx);

  for (x = 0; x < dims[1]; x++)
    max_filter_1d(tmpM+x, PyArray_GETPTR2(pydeformed, 0, x), tmpIy+x, s, dims[1], dims[0], ay, by);

  free(tmpM);
  free(tmpIx);
  free(tmpIy);

  return PyArray_Return(pydeformed);
}

PyObject * filter_image (PyArrayObject * pyfeatures, PyArrayObject * pyfilter, float bias) {
    npy_intp * features_dims = PyArray_DIMS(pyfeatures);
    npy_intp * filter_dims = PyArray_DIMS(pyfilter);
    int top_pad = (filter_dims[0]-1)/2;
    int bottom_pad = filter_dims[0] - 1 - top_pad; 
    int left_pad = (filter_dims[1]-1)/2;
    int right_pad = filter_dims[1] - 1 - left_pad; 
    int a, b, l;
    PyArrayObject * pyfiltered = NULL;
    npy_intp * features_stride = PyArray_STRIDES(pyfeatures);
    npy_intp * filtered_stride = NULL;
    npy_intp filtered_dims[2] = {0, 0};

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

    filtered_dims[0] = features_dims[0];
    filtered_dims[1] = features_dims[1];
    pyfiltered = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, filtered_dims, NPY_FLOAT);

    filtered_stride = PyArray_STRIDES(pyfiltered);

    /* zero out array */
    for (a = 0; a < filtered_dims[0]; ++a) {
        for (b = 0; b < filtered_dims[1]; ++b) {
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -bias;
        }
    }

    /* for each layer */
    for (l = 0; l < 32; ++l) {
        /* iterate over filter which should be tiny compared to the image */
        int i;
        for (i = 0; i < filter_dims[0]; ++i) {
            int j;
            for (j = 0; j < filter_dims[1]; ++j) {
                float weight = *(float*)PyArray_GETPTR3(pyfilter, i, j, l);
                int k;
                for (k = 0; k < features_dims[0]-filter_dims[0]+1; ++k) {
                    float * out = PyArray_GETPTR2(pyfiltered, k+(filter_dims[0]-1)/2, (filter_dims[1]-1)/2);
                    float * in = PyArray_GETPTR3(pyfeatures, i+k, j, l);
                    cblas_saxpy(features_dims[1]-filter_dims[1]+1, weight, in, features_stride[1]/sizeof(float), out, filtered_stride[1]/sizeof(float));
                }
            }
        }
    }

    /* invalidate edges */
    for (a = 0; a < min(filtered_dims[0], top_pad); ++a)
        for (b = 0; b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;

    for (a = max(0, filtered_dims[0]-bottom_pad); a < filtered_dims[0]; ++a)
        for (b = 0; b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;

    for (a = 0; a < filtered_dims[0]; ++a)
        for (b = 0; b < min(filtered_dims[1], left_pad); ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;

    for (a = 0; a < filtered_dims[0]; ++a)
        for (b = max(0, filtered_dims[1]-right_pad); b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -INFINITY;

    return PyArray_Return(pyfiltered);
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
    if (!PyArg_ParseTuple(args, "O!O!|f", &PyArray_Type, &pyfeatures, &PyArray_Type, &pyfilter, &bias)) 
        return NULL;
    return filter_image(pyfeatures, pyfilter, bias);
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
