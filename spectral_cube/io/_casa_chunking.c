#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Fast histogram functioins";
static char _combine_chunks_docstring[] = "Compute a 1D histogram";

/* Declare the C functions here. */
static PyObject *_combine_chunks(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_combine_chunks", _combine_chunks, METH_VARARGS, _combine_chunks_docstring},
    {NULL, NULL, 0, NULL}};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods)     \
    static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT,              \
        name,                               \
        doc,                                \
        -1,                                 \
        methods,                            \
    };                                      \
    ob = PyModule_Create(&moduledef);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_DEF(ob, name, doc, methods) \
    ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_casa_chunking)
{
    PyObject *m;
    MOD_DEF(m, "_casa_chunking", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

static PyObject *_combine_chunks(PyObject *self, PyObject *args)
{

    long n;
    double xmin, xmax, tx, fnx, normx;
    PyObject *input_obj, *output_obj;
    PyArrayObject *input_array, *output_array;
    int ox, oy, oz, ow, nx, ny, nz, nw, nsx, nsy, nsz, nsw;
    npy_intp dims[1];
    double *output;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr, index;
    PyArray_Descr *dtype;
    int bx, by, bz, bw, i, j, k, l, i_o, j_o, k_o, l_o, i_f, j_f, k_f, l_f;

    // NOTE: this function is written in a way to work with 4-d data as it can
    // then easily be called with 3-d data and have a dimension removed.

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oiiiiiiii", &input_obj, &nx, &ny, &nz, &nw, &ox, &oy, &oz, &ow))
    {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    input_array = (PyArrayObject *)PyArray_FROM_O(input_obj);

    /* If that didn't work, throw an `Exception`. */
    if (input_array == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input array.");
        Py_XDECREF(input_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(input_array, 0);

    /* Build the output array */
    dims[0] = n;
    output_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_obj == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
        Py_DECREF(input_array);
        Py_XDECREF(output_obj);
        return NULL;
    }

    output_array = (PyArrayObject *)output_obj;

    //  TODO: the following is probably unecessary
    PyArray_FILLWBYTE(output_array, 0);

    if (n == 0)
    {
        Py_DECREF(input_array);
        return output_obj;
    }

    // TODO: can probaby assume data is contiguous and might be able to
    // therefore simplify some of the folloing
    dtype = PyArray_DescrFromType(NPY_DOUBLE);
    iter = NpyIter_New(input_array,
                       NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                       NPY_KEEPORDER, NPY_SAFE_CASTING, dtype);
    if (iter == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
        Py_DECREF(input_array);
        Py_DECREF(output_obj);
        Py_DECREF(output_array);
        return NULL;
    }

    /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
        NpyIter_Deallocate(iter);
        Py_DECREF(input_array);
        Py_DECREF(output_obj);
        Py_DECREF(output_array);
        return NULL;
    }

    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);

    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);

    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* Get C array for output array */
    output = (double *)PyArray_DATA(output_array);

    Py_BEGIN_ALLOW_THREADS

    npy_intp stride = *strideptr;
    npy_intp size = *innersizeptr;

    for (bw = 0; bw < ow; ++bw)
    {
        l_o = bw * nw;
        for (bz = 0; bz < oz; ++bz)
        {
            k_o = bz * nz;
            for (by = 0; by < oy; ++by)
            {
                j_o = by * ny;
                for (bx = 0; bx < ox; ++bx)
                {
                    i_o = bx * nx;
                    for (l = 0; l < nw; ++l)
                    {
                        for (k = 0; k < nz; ++k)
                        {
                            for (j = 0; j < ny; ++j)
                            {
                                for (i = 0; i < nx; ++i)
                                {

                                    if (size == 0)
                                    {
                                        iternext(iter);
                                        stride = *strideptr;
                                        size = *innersizeptr;
                                    }

                                    i_f = i_o + i;
                                    j_f = j_o + j;
                                    k_f = k_o + k;
                                    l_f = l_o + l;

                                    index = i_f + j_f * (nx * ox) + k_f * (nx * ox * ny * oy) + l_f * (nx * ox * ny * oy * nz * oz);

                                    output[index] = *(double *)dataptr[0];
                                    dataptr[0] += stride;

                                    size--;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    NpyIter_Deallocate(iter);

    /* Clean up. */
    Py_DECREF(input_array);

    return output_obj;
}
