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

// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
static float uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
static float vv[9] = {0.0000, 
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
PyObject *process(PyArrayObject *pyimage, const int sbin) {
  const npy_intp *dims = PyArray_DIMS(pyimage);
  int cells[2];
  int visible[2];
  npy_intp out[3];
  float *hist = NULL;
  float *norm = NULL;
  PyArrayObject *pyfeat = NULL;
  int x, y;  
  int o;

  if (PyArray_NDIM(pyimage) != 3 ||
      dims[2] != 3 ||
      PyArray_DESCR(pyimage)->type_num != NPY_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Array must be a float precision 3 channel color image");
    return NULL;
  }

  // memory for caching orientation histograms & their norms
  cells[0] = (int)round((float)dims[0]/(float)sbin);
  cells[1] = (int)round((float)dims[1]/(float)sbin);
  hist = (float*)calloc(cells[0]*cells[1]*18, sizeof(float));
  norm = (float *)calloc(cells[0]*cells[1], sizeof(float));

  // memory for HOG features
  out[0] = maxi(cells[0]-2, 0);
  out[1] = maxi(cells[1]-2, 0);
  out[2] = 27+4+1;
  pyfeat = (PyArrayObject*)PyArray_SimpleNew((npy_intp)3, out, NPY_FLOAT);
  
  visible[0] = cells[0]*sbin;
  visible[1] = cells[1]*sbin;

  for (x = 1; x < visible[1]-1; x++) {
    for (y = 1; y < visible[0]-1; y++) {
      int xpos = mini(x, dims[1]-2);
      int ypos = mini(y, dims[0]-2);

      // first color channel
      float s = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 0);
      float dy = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 0);
      float dx = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 0);
      float v = dx*dx + dy*dy;

      float dy2, dx2, v2, dy3, dx3, v3;
      float best_dot = 0;
      int best_o = 0;
      float xp, yp, vx0, vy0, vx1, vy1;
      int ixp, iyp; 
       
      // second color channel
      s += *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 1);
      dy2 = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 1);
      dx2 = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 1);
      v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += *(float*)PyArray_GETPTR3(pyimage, ypos, xpos, 2);
      dy3 = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 2) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 2);
      dx3 = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 2) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 2);
      v3 = dx3*dx3 + dy3*dy3;

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
      xp = ((float)x+0.5)/(float)sbin - 0.5;
      yp = ((float)y+0.5)/(float)sbin - 0.5;
      ixp = (int)floor(xp);
      iyp = (int)floor(yp);
      vx0 = xp-ixp;
      vy0 = yp-iyp;
      vx1 = 1.0-vx0;
      vy1 = 1.0-vy0;
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
      float t1 = 0;
      float t2 = 0;
      float t3 = 0;
      float t4 = 0;

      p = norm + (x+1)*cells[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + (x+1)*cells[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + x*cells[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);
      p = norm + x*cells[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+cells[0]) + *(p+cells[0]+1) + eps);

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
  return PyArray_Return(pyfeat);
}

static PyObject * ComputeFeatures(PyObject * self, PyObject * args)
{
    PyArrayObject * pyimage;
    int sbin;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pyimage, &sbin)) 
        return NULL;
    return process(pyimage, sbin);
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_features",
    "Pydro features native implementation.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION < 3
static PyMethodDef _features_methods[] = {
    {"ComputeFeatures", ComputeFeatures, METH_VARARGS, "Compute Pedro's special HoG features."},
    {NULL}
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__features(void)
#else
PyMODINIT_FUNC
init_features(void)
#endif
{
    import_array();

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    Py_InitModule3("_features", _features_methods, "Pydro native feature implementation.");
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
