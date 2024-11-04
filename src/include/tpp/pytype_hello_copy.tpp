#ifndef __PYTYPE_HELLO_COPY_TPP__
#define __PYTYPE_HELLO_COPY_TPP__


static PyObject*
Hello_copy(HelloObject* self, PyObject* args, PyObject* kwds)
{
    std::vector<std::vector<std::vector<std::vector<std::int64_t> > > > Tensor;

    // Inputs as keywords list
    static char *kwlist[]       = {"input", NULL};

    PyObject      *_PyFT = NULL;
    PyArrayObject *_PyArrayFT = NULL;
    PyArrayObject *_PyArrayOT = NULL;
    char* logfile      = NULL;

    // Parse inputs as tuples or keywords list
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &_PyFT)) {
        PyErr_SetString(MyModuleError, "Cannot parse input");
        return NULL;
    }

    // Check Inputs
    if(!_PyFT) {
        PyErr_SetString(MyModuleError, "No FT input array parsed");
        return NULL;
    }
    _PyArrayFT = (PyArrayObject*) PyArray_FROM_OF(_PyFT, NPY_ARRAY_IN_ARRAY);
    if(!_PyArrayFT) {
        PyErr_SetString(MyModuleError, "Parsed FT input is not contiguos and aligned");
        return NULL;
    }

    // Check Input Tensors
    if(PyArray_NDIM(_PyArrayFT) != 4) {
        PyErr_SetString(MyModuleError, "Cannot copy FT PyArray to C++ tensor (unsupported ndim)");
        return NULL;
    }
    if(PyArray_TYPE(_PyArrayFT) != NPY_INT64) {
        PyErr_SetString(MyModuleError, "Input FT is not INT64");
        return NULL;
    }

    // Copy PyArray to C++ Vector
    if(!INumPy::_PyArray_to_4Dvector<std::int64_t>(_PyArrayFT, Tensor)) {
        PyErr_SetString(MyModuleError, "Cannot copy FT PyArray to C++ tensor (4D tensor)");
        return NULL;
    }

    //* ---     RETURN     --- *//
    if(!INumPy::_4Dvector_to_PyArray<std::int64_t>(Tensor, &_PyArrayOT)) {
        PyErr_SetString(MyModuleError, "Cannot copy Out C++ tensor to PyArray (4D tensor)");
        return NULL;
    }

    Py_XDECREF(_PyArrayFT);

    return PyArray_Return(_PyArrayOT);
}

#endif