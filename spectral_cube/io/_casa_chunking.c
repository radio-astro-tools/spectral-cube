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
    PyObject *input_obj, *output_obj;
    PyArrayObject *input_array, *output_array;
    int ox, oy, oz, ow, nx, ny, nz, nw;
    npy_intp dims[1];
    double *output;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strideptr, *innersizeptr, index_in, index_out;
    PyArray_Descr *dtype;
    int bx, by, bz, bw, i, j, k, l, i_o, j_o, k_o, l_o, i_f, j_f, k_f, l_f, itemsize;

    // NOTE: this function is written in a way to work with 4-d data as it can
    // then easily be called with 3-d data and have a dimension removed. We also
    // take a byte array as input, assume the data is contiguous, and take the size
    // of each elements in bytes - this allows us to reuse the same function for
    // mask and data of different types.

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oiiiiiiiii", &input_obj, &itemsize, &nx, &ny, &nz, &nw, &ox, &oy, &oz, &ow))
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

    /* How many bytes are there? */
    n = (long)PyArray_DIM(input_array, 0);

    /* Build the output array */
    dims[0] = n;
    output_obj = PyArray_SimpleNew(1, dims, NPY_UINT8);
    if (output_obj == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
        Py_DECREF(input_array);
        Py_XDECREF(output_obj);
        return NULL;
    }

    output_array = (PyArrayObject *)output_obj;

    if (n == 0)
    {
        Py_DECREF(input_array);
        return output_obj;
    }

    /* Get C array for input and output arrays */
    input = (uint8_t *)PyArray_DATA(input_array);
    output = (uint8_t *)PyArray_DATA(output_array);

    Py_BEGIN_ALLOW_THREADS

    index_in = 0;

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

                                    i_f = i_o + i;
                                    j_f = j_o + j;
                                    k_f = k_o + k;
                                    l_f = l_o + l;

                                    index_out = (i_f + j_f * (nx * ox) + k_f * (nx * ox * ny * oy) + l_f * (nx * ox * ny * oy * nz * oz)) * itemsize;

                                    for (itempos=0; itempos<itemsize;++itempos) {
                                        output[index_out + itempos] = input[index_in];
                                        index_in++;
                                    }

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
