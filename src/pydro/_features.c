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
PyObject *process(PyArrayObject *pyimage, const int sbin, const int pad_x, const int pad_y) {
  const npy_intp *dims = PyArray_DIMS(pyimage);
  int cells[2];
  int visible[2];
  npy_intp out[3];
  float *hist = NULL;
  float *norm = NULL;
  PyArrayObject *pyfeat = NULL;
  int x, y, l;  
  int o;

  if (PyArray_NDIM(pyimage) != 3 ||
      dims[2] != 3 ||
      PyArray_DESCR(pyimage)->type_num != NPY_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Array must be a float precision 3 channel color image");
    return NULL;
  }

  // memory for caching orientation histograms & their norms
  cells[0] = (int)round((float)dims[0]/sbin);
  cells[1] = (int)round((float)dims[1]/sbin);
  hist = (float*)calloc(cells[0]*cells[1]*18, sizeof(float));
  norm = (float *)calloc(cells[0]*cells[1], sizeof(float));

  // memory for HOG features
  out[0] = maxi(cells[0]-2, 0)+2*pad_y;
  out[1] = maxi(cells[1]-2, 0)+2*pad_x;
  out[2] = 27+4+1;

  npy_intp strides [3] = { out[1]*out[2]*sizeof(float), sizeof(float), out[1]*sizeof(float) };
  pyfeat = (PyArrayObject*)PyArray_New(
    &PyArray_Type, (npy_intp)3, out, NPY_FLOAT,
    strides, NULL, 0, 0, NULL
  );
  
  visible[0] = cells[0]*sbin;
  visible[1] = cells[1]*sbin;

  for (x = 1; x < visible[1]-1; x++) {
    for (y = 1; y < visible[0]-1; y++) {
      int xpos = mini(x, dims[1]-2);
      int ypos = mini(y, dims[0]-2);

      // first color channel
      float dy = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 0);
      float dx = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 0) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 0);
      float v = dx*dx + dy*dy;

      float dy2, dx2, v2, dy3, dx3, v3;
      float best_dot = 0;
      int best_o = 0;
      float xp, yp, vx0, vy0, vx1, vy1;
      int ixp, iyp; 
       
      // second color channel
      dy2 = *(float*)PyArray_GETPTR3(pyimage, ypos+1, xpos, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos-1, xpos, 1);
      dx2 = *(float*)PyArray_GETPTR3(pyimage, ypos, xpos+1, 1) - *(float*)PyArray_GETPTR3(pyimage, ypos, xpos-1, 1);
      v2 = dx2*dx2 + dy2*dy2;

      // third color channel
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
      xp = (x+0.5)/(float)sbin - 0.5;
      yp = (y+0.5)/(float)sbin - 0.5;
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
  for (x = 0; x < out[1]-pad_x*2; x++) {
    for (y = 0; y < out[0]-pad_y*2; y++) {
      float *src, *p, n1, n2, n3, n4;
      float t1 = 0.0;
      float t2 = 0.0;
      float t3 = 0.0;
      float t4 = 0.0;

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
        *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, o) = 0.5 * (h1 + h2 + h3 + h4);
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
        *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, o+18) = 0.5 * (h1 + h2 + h3 + h4);
        src += cells[0]*cells[1];
      }

      // texture features
      *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, 27) = 0.2357 * t1;
      *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, 28) = 0.2357 * t2;
      *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, 29) = 0.2357 * t3;
      *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, 30) = 0.2357 * t4;

      // truncation feature
      *(float*)PyArray_GETPTR3(pyfeat, y+pad_y, x+pad_x, 31) = 0.0;
    }
  }

  for (y = 0; y < out[0]; ++y) {
      for (x = 0; x < pad_x; ++x) {
        int x_op = out[1] - x - 1;
        for (l = 0; l < 31; ++l) {
            *(float*)PyArray_GETPTR3(pyfeat, y, x, l) = 0;

            *(float*)PyArray_GETPTR3(pyfeat, y, x_op, l) = 0;
        }
        *(float*)PyArray_GETPTR3(pyfeat, y, x, 31) = 1;
        *(float*)PyArray_GETPTR3(pyfeat, y, x_op, 31) = 1;
      }
  }
  for (x = 0; x < out[1]; ++x) {
      for (y = 0; y < pad_y; ++y) {
        int y_op = out[0] - y - 1;
        for (l = 0; l < 31; ++l) {
            *(float*)PyArray_GETPTR3(pyfeat, y, x, l) = 0;
            *(float*)PyArray_GETPTR3(pyfeat, y_op, x, l) = 0;
        }
        *(float*)PyArray_GETPTR3(pyfeat, y, x, 31) = 1;

        *(float*)PyArray_GETPTR3(pyfeat, y_op, x, 31) = 1;
      }
  }


  free(hist);
  free(norm);
  return PyArray_Return(pyfeat);
}

/*
 * Fast image subsampling.
 * This is used to construct the feature pyramid.
 */

// struct used for caching interpolation values
struct alphainfo {
  int si, di;
  float alpha;
};

// copy src into dst using pre-computed interpolation values
void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n) {
  struct alphainfo *end = ofs + n;
  while (ofs != end) {
    dst[ofs->di] += ofs->alpha * src[ofs->si];
    ofs++;
  }
}

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(float *src, int sheight, float *dst, int dheight, 
		  int width, int chan) {
  float scale = (float)dheight/(float)sheight;
  float invscale = (float)sheight/(float)dheight;
  
  // we cache the interpolation values since they can be 
  // shared among different columns
  int len = (int)ceil(dheight*invscale) + 2*dheight;
  struct alphainfo ofs[len];
  int k = 0;
  int dy, sy, c, x;
  for (dy = 0; dy < dheight; dy++) {
    float fsy1 = dy * invscale;
    float fsy2 = fsy1 + invscale;
    int sy1 = (int)ceil(fsy1);
    int sy2 = (int)floor(fsy2);       

    if (sy1 - fsy1 > 1e-3) {
      assert(k < len);
      ofs[k].di = chan*dy;
      ofs[k].si = chan*(sy1-1)*width;
      ofs[k++].alpha = (sy1 - fsy1) * scale;
    }

    for (sy = sy1; sy < sy2; sy++) {
      assert(k < len);
      assert(sy < sheight);
      ofs[k].di = chan*dy;
      ofs[k].si = chan*sy*width;
      ofs[k++].alpha = scale;
    }

    if (fsy2 - sy2 > 1e-3) {
      assert(k < len);
      assert(sy2 < sheight);
      ofs[k].di = chan*dy;
      ofs[k].si = chan*sy2*width;
      ofs[k++].alpha = (fsy2 - sy2) * scale;
    }
  }

  // resize each column of each color channel
  bzero(dst, chan*width*dheight*sizeof(float));
  for (c = 0; c < chan; c++) {
    for (x = 0; x < width; x++) {
      float *s = src + c + chan*x;
      float *d = dst + c + chan*x*dheight;
      alphacopy(s, d, ofs, k);
    }
  }
}

PyObject *resize_image(PyArrayObject * pyimage, int y, int x) {
  npy_intp * sdims = PyArray_DIMS(pyimage);
  npy_intp * strides = PyArray_STRIDES(pyimage);
  npy_intp ddims[3];
  PyArrayObject * pyresized = NULL;

  if (PyArray_NDIM(pyimage) != 3) {
    PyErr_SetString(PyExc_TypeError, "Input image must be three channels.");
    return NULL;
  }

  if (PyArray_DESCR(pyimage)->type_num != NPY_FLOAT) {
    PyErr_SetString(PyExc_TypeError, "Input image must be floating point.");
    return NULL;
  }

  if (sdims[2] != 3) {
    PyErr_SetString(PyExc_TypeError, "Input image third dimension is wrong size.");
    return NULL;
  }

  if (strides[0] != sdims[1]*sdims[2]*sizeof(float) || strides[1] != sdims[2]*sizeof(float) || strides[2] != sizeof(float)) {
    PyErr_SetString(PyExc_TypeError, "Unexpected strides");
    return NULL;
  }

  ddims[0] = y;
  ddims[1] = x;
  ddims[2] = sdims[2];

  pyresized = (PyArrayObject*)PyArray_SimpleNew((npy_intp)3, ddims, NPY_FLOAT);

  float *tmp = (float*)calloc(ddims[0]*sdims[1]*sdims[2], sizeof(float));
  resize1dtran((float*)PyArray_DATA(pyimage), sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
  resize1dtran(tmp, sdims[1], (float*)PyArray_DATA(pyresized), ddims[1], ddims[0], sdims[2]);

  free(tmp);

  return PyArray_Return(pyresized);
}


static PyObject * ComputeFeatures(PyObject * self, PyObject * args)
{
    PyArrayObject * pyimage;
    int sbin, pad_x, pad_y;
    if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &pyimage, &sbin, &pad_x, &pad_y)) 
        return NULL;
    return process(pyimage, sbin, pad_x, pad_y);
}

static PyObject * ResizeImage (PyObject * self, PyObject * args)
{
    PyArrayObject * pyimage;
    int x, y;
    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &pyimage, &y, &x)) {
        return NULL;
    }
    return resize_image (pyimage, y, x);
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
    {"ResizeImage", ResizeImage, METH_VARARGS, "Resize image using Pedro's fast implementation."},
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
