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
double uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
double vv[9] = {0.0000, 
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
// takes a double color image and a bin size 
// returns HOG features
PyArrayObject *process(PyArrayObject *pyimage, const int sbin) {
  const npy_intp *dims = PyArray_DIMS(pyimage);
  if (PyArray_NDIM(pyimage) != 3 ||
      dims[2] != 3 ||
      PyArray_DESCR(pyimage)->type_num != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Array must be a double precision 3 channel color image");
    return NULL;
  }

  // memory for caching orientation histograms & their norms
  int cells[2];
  cells[0] = (int)round((double)dims[0]/(double)sbin);
  cells[1] = (int)round((double)dims[1]/(double)sbin);
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
      double s = *(double*)PyArray_GETPTR3(pyimage, ypos, xpos, 0);
      double dy = *(double*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 0) - *(double*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 0);
      double dx = *(double*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 0) - *(double*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 0);
      double v = dx*dx + dy*dy;

      // second color channel
      s += *(double*)PyArray_GETPTR3(pyimage, ypos, xpos, 1);
      double dy2 = *(double*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 1) - *(double*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 1);
      double dx2 = *(double*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 1) - *(double*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 1);
      double v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += *(double*)PyArray_GETPTR3(pyimage, ypos, xpos, 2);
      double dy3 = *(double*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 2) - *(double*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 2);
      double dx3 = *(double*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 2) - *(double*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 2);
      double v3 = dx3*dx3 + dy3*dy3;

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
      double best_dot = 0;
      int best_o = 0;
      int o;
      for (o = 0; o < 9; o++) {
        double dot = uu[o]*dx + vv[o]*dy;
        if (dot > best_dot) {
          best_dot = dot;
          best_o = o;
        } else if (-dot > best_dot) {
          best_dot = -dot;
          best_o = o+9;
        }
      }
      
      // add to 4 histograms around pixel using bilinear interpolation
      double xp = ((double)x+0.5)/(double)sbin - 0.5;
      double yp = ((double)y+0.5)/(double)sbin - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      double vx0 = xp-ixp;
      double vy0 = yp-iyp;
      double vx1 = 1.0-vx0;
      double vy1 = 1.0-vy0;
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

static PyObject * pydro_ComputeFeatures(PyObject * self, PyObject * args)
{
    PyArrayObject * pyimage;
    int sbin;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pyimage, &sbin)) {
        PyErr_SetString(PyExc_TypeError, "ComputeFeatures requires a 3 channel color double precision image and an integer.");
        return NULL;
    }
    return (PyObject*)process(pyimage, sbin);
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
