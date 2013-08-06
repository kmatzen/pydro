// AUTORIGHTS
// -------------------------------------------------------
// Copyright (C) 2011-2012 Ross Girshick
// Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
// Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan
// 
// This file is part of the voc-releaseX code
// (http://people.cs.uchicago.edu/~rbg/latent/)
// and is available under the terms of an MIT-like license
// provided in COPYING. Please retain this notice and
// COPYING if you use this file (or a portion of it) in
// your project.
// -------------------------------------------------------

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>

#include <mkl_cblas.h>

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
float uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
float vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

static inline float minf(float x, float y) { return (x <= y ? x : y); }
static inline float maxf(float x, float y) { return (x <= y ? y : x); }

static inline int mini(int x, int y) { return (x <= y ? x : y); }
static inline int maxi(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a float color image and a bin size 
// returns HOG features
PyArrayObject *process(PyArrayObject *pyimage, const int sbin) {
  const npy_intp *dims = PyArray_DIMS(pyimage);
  if (PyArray_NDIM(pyimage) != 3 ||
      dims[2] != 3 ||
      PyArray_DESCR(pyimage)->type_num != NPY_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Array must be a float precision 3 channel color image");
    return NULL;
  }

  // memory for caching orientation histograms & their norms
  int cells[2];
  cells[0] = (int)round((float)dims[0]/(float)sbin);
  cells[1] = (int)round((float)dims[1]/(float)sbin);
  float *hist = (float*)calloc(cells[0]*cells[1]*18, sizeof(float));
  float *norm = (float *)calloc(cells[0]*cells[1], sizeof(float));

  // memory for HOG features
  npy_intp out[3];
  out[0] = maxi(cells[0]-2, 0);
  out[1] = maxi(cells[1]-2, 0);
  out[2] = 27+4+1;
  PyArrayObject *pyfeat = (PyArrayObject*)PyArray_SimpleNew((npy_intp)3, out, NPY_FLOAT);
  
  int visible[2];
  visible[0] = cells[0]*sbin;
  visible[1] = cells[1]*sbin;

  int x, y;  
  for (x = 1; x < visible[1]-1; x++) {
    for (y = 1; y < visible[0]-1; y++) {
      int xpos = mini(x, dims[1]-2);
      int ypos = mini(y, dims[0]-2);

      // first color channel
      float s = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 0);
      float dy = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 0);
      float dx = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 0);
      float v = dx*dx + dy*dy;

      // second color channel
      s += *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 1);
      float dy2 = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 1);
      float dx2 = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 1);
      float v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 2);
      float dy3 = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 2) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 2);
      float dx3 = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 2) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 2);
      float v3 = dx3*dx3 + dy3*dy3;

      // pick channel with strongest gradient
      if (v2 > v) {
        v = v2;
        dx = dx2;
        dy = dy2;
      } 
      if (v3 > v) {
        v = v3;
        dx = dx3;
        dy = dy3;
      }

      // snap to one of 18 orientations
      float best_dot = 0;
      int best_o = 0;
      int o;
      for (o = 0; o < 9; o++) {
        float dot = uu[o]*dx + vv[o]*dy;
        if (dot > best_dot) {
          best_dot = dot;
          best_o = o;
        } else if (-dot > best_dot) {
          best_dot = -dot;
          best_o = o+9;
        }
      }
      
      // add to 4 histograms around pixel using bilinear interpolation
      float xp = ((float)x+0.5)/(float)sbin - 0.5;
      float yp = ((float)y+0.5)/(float)sbin - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      float vx0 = xp-ixp;
      float vy0 = yp-iyp;
      float vx1 = 1.0-vx0;
      float vy1 = 1.0-vy0;
      v = sqrt(v);

      if (ixp >= 0 && iyp >= 0) {
        *(hist + ixp*cells[0] + iyp + best_o*cells[0]*cells[1]) += 
          vx1*vy1*v;
      }

      if (ixp+1 < cells[1] && iyp >= 0) {
        *(hist + (ixp+1)*cells[0] + iyp + best_o*cells[0]*cells[1]) += 
          vx0*vy1*v;
      }

      if (ixp >= 0 && iyp+1 < cells[0]) {
        *(hist + ixp*cells[0] + (iyp+1) + best_o*cells[0]*cells[1]) += 
          vx1*vy0*v;
      }

      if (ixp+1 < cells[1] && iyp+1 < cells[0]) {
        *(hist + (ixp+1)*cells[0] + (iyp+1) + best_o*cells[0]*cells[1]) += 
          vx0*vy0*v;
      }
    }
  }

  // compute energy in each block by summing over orientations
  int o;
  for (o = 0; o < 9; o++) {
    float *src1 = hist + o*cells[0]*cells[1];
    float *src2 = hist + (o+9)*cells[0]*cells[1];
    float *dst = norm;
    float *end = norm + cells[1]*cells[0];
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (x = 0; x < out[1]; x++) {
    for (y = 0; y < out[0]; y++) {
      float *src, *p, n1, n2, n3, n4;

      p = norm + (x+1)*cells[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + (x+1)*cells[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + x*cells[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + x*cells[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);

      float t1 = 0;
      float t2 = 0;
      float t3 = 0;
      float t4 = 0;

      // contrast-sensitive features
      src = hist + (x+1)*cells[0] + (y+1);
      for (o = 0; o < 18; o++) {
        float h1 = minf(*src * n1, 0.2);
        float h2 = minf(*src * n2, 0.2);
        float h3 = minf(*src * n3, 0.2);
        float h4 = minf(*src * n4, 0.2);
        *(float*)PyArray_GETPTR3(pyfeat, y, x, o) = 0.5 * (h1 + h2 + h3 + h4);
        t1 += h1;
        t2 += h2;
        t3 += h3;
        t4 += h4;
        src += cells[0]*cells[1];
      }

      // contrast-insensitive features
      src = hist + (x+1)*cells[0] + (y+1);
      for (o = 0; o < 9; o++) {
        float sum = *src + *(src + 9*cells[0]*cells[1]);
        float h1 = minf(sum * n1, 0.2);
        float h2 = minf(sum * n2, 0.2);
        float h3 = minf(sum * n3, 0.2);
        float h4 = minf(sum * n4, 0.2);
        *(float*)PyArray_GETPTR3(pyfeat, y, x, o+18) = 0.5 * (h1 + h2 + h3 + h4);
        src += cells[0]*cells[1];
      }

      // texture features
      *(float*)PyArray_GETPTR3(pyfeat, y, x, 27) = 0.2357 * t1;
      *(float*)PyArray_GETPTR3(pyfeat, y, x, 28) = 0.2357 * t2;
      *(float*)PyArray_GETPTR3(pyfeat, y, x, 29) = 0.2357 * t3;
      *(float*)PyArray_GETPTR3(pyfeat, y, x, 30) = 0.2357 * t4;

      // truncation feature
      *(float*)PyArray_GETPTR3(pyfeat, y, x, 31) = 0;
    }
  }

  free(hist);
  free(norm);
  return pyfeat;
}

PyArrayObject * detect (PyArrayObject * pyfeatures, PyArrayObject * pyfilter, float bias) {
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

    npy_intp * features_dims = PyArray_DIMS(pyfeatures);
    npy_intp * filter_dims = PyArray_DIMS(pyfilter);

    if (features_dims[2] != 32) {
        PyErr_SetString(PyExc_TypeError, "features' feature dimsionality should be 32.");
        return NULL;
    }

    if (filter_dims[2] != 32) {
        PyErr_SetString(PyExc_TypeError, "filters' feature dimensionality should be 32.");
        return NULL;
    }

    npy_intp filtered_dims[2];
    filtered_dims[0] = features_dims[0];
    filtered_dims[1] = features_dims[1];
    PyArrayObject * pyfiltered = (PyArrayObject*)PyArray_SimpleNew((npy_intp)2, filtered_dims, NPY_FLOAT);

    npy_intp * features_stride = PyArray_STRIDES(pyfeatures);
    npy_intp * filtered_stride = PyArray_STRIDES(pyfiltered);

    // zero out array
    int a;
    int b;
    for (a = 0; a < filtered_dims[0]; ++a) {
        for (b = 0; b < filtered_dims[1]; ++b) {
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -bias;
        }
    }

    // for each layer
    int l;
    for (l = 0; l < 32; ++l) {
        // iterate over filter which should be tiny compared to the image
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

    // invalidate edges
    int top_pad = (filter_dims[0]-1)/2;
    for (a = 0; a < top_pad+1; ++a)
        for (b = 0; b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -2;

    int bottom_pad = filter_dims[0] - 1 - top_pad; 
    for (a = filtered_dims[0]-bottom_pad-1; a < filtered_dims[0]; ++a)
        for (b = 0; b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -2;

    int left_pad = (filter_dims[1]-1)/2;
    for (a = 0; a < filtered_dims[0]; ++a)
        for (b = 0; b < left_pad+1; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -2;

    int right_pad = filter_dims[1] - 1 - left_pad; 
    for (a = 0; a < filtered_dims[0]; ++a)
        for (b = filtered_dims[1]-right_pad-1; b < filtered_dims[1]; ++b)
            *(float*)PyArray_GETPTR2(pyfiltered, a, b) = -2;

    return pyfiltered;
}

static PyObject * pydro_ComputeFeatures(PyObject * self, PyObject * args)
{
    PyArrayObject * pyimage;
    int sbin;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pyimage, &sbin)) 
        return NULL;
    return (PyObject*)process(pyimage, sbin);
}

static PyObject * pydro_Detect(PyObject * self, PyObject * args)
{
    PyArrayObject * pyfeatures;
    PyArrayObject * pyfilter;
    float bias = 0.0f;
    if (!PyArg_ParseTuple(args, "O!O!|f", &PyArray_Type, &pyfeatures, &PyArray_Type, &pyfilter, &bias)) 
        return NULL;
    return (PyObject*)detect(pyfeatures, pyfilter, bias);
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pydro",
    "Wrapper for Pedro Felzenszwalb's MATLAB implementations.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef pydro_methods[] = {
    {"ComputeFeatures", pydro_ComputeFeatures, METH_VARARGS, "Compute Pedro's special HoG features."},
    {"Detect", pydro_Detect, METH_VARARGS, "Compute a 2D cross correlation between a filter and image features.  Optionally add bias term."},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_pydro(void)
#else
PyMODINIT_FUNC
initpydro(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("pydro", pydro_methods, "Wrapper for Pedro Felzenszwalb's MATLAB implementations.");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
