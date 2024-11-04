
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>

#include "mymodule_error.h"

#include "inumpy.hpp"
#include "pytype_hello.h"


//* === Module Definition === *//
static PyModuleDef MyModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "MyModule",
    .m_doc    = "Example C-extension Module",
    .m_size = -1,
};


//* === Init Functions === *//
PyMODINIT_FUNC
PyInit_MyModule(void)
{
    PyObject *m;
    import_array();

    // Check HelloType
    if(PyType_Ready(&HelloType) < 0) { return NULL; }

    // Create MyModule
    m = PyModule_Create(&MyModule);
    if(m == NULL) { return NULL; }

    // Allocate Custom Type
    Py_INCREF(&HelloType);
    if (PyModule_AddObject(m, "hello", (PyObject*) &HelloType) < 0) {
        Py_DECREF(&HelloType);
        Py_DECREF(m);
        return NULL;
    }

    // Allocate Custom Error Handler
    MyModuleError = PyErr_NewException("MyModule.error", NULL, NULL);
    Py_XINCREF(MyModuleError);
    if(PyModule_AddObject(m, "error", MyModuleError) < 0) {
        Py_XDECREF(MyModuleError);
        Py_CLEAR(MyModuleError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
