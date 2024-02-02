
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
//#include <arrayobject.h>

#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>

#include "c_methods.h"

//* === Python Methods === *//
static PyMethodDef MyModuleMethods[] = {
  {"hello", C_hello, METH_VARARGS, "Performs a helloolution on the NVDLA EMU"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef MyModule = {
  PyModuleDef_HEAD_INIT,
  "MyModule",    // name of the module
  NULL,           // module documentation (we have not)
  -1,             // Size of per-interpreter module (keeps state in global vars)
  MyModuleMethods // PyMethods struct
};



//* === Init Functions === *//
PyMODINIT_FUNC
PyInit_MyModule(void)
{
  PyObject *m;

//  import_array();

  m = PyModule_Create(&MyModule);
  if(m == NULL) return NULL;

  MyModuleError = PyErr_NewException("MyModule.error", NULL, NULL);
  Py_XINCREF(MyModuleError);
  if(PyModule_AddObject(m, "error", MyModuleError)<0) {
    Py_XDECREF(MyModuleError);
    Py_CLEAR(MyModuleError);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}

int main(int argc, char* argv[])
{
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if(program == NULL) {
    fprintf(stderr, "Fatal eror: cannot decode argv[0]\n");
    exit(1);
  }

  // Add built-in module before init
  if(PyImport_AppendInittab("MyModule", PyInit_MyModule) == -1) {
    fprintf(stderr, "Error: could not extend in-built modules table\n");
    exit(1);
  }

  // Pass argv[0]
  Py_SetProgramName(program);

  // Initialize Python Interpreter
  Py_Initialize();

  // Import Module
  PyObject *pmodule = PyImport_ImportModule("MyModule");
  if(!pmodule) {
    PyErr_Print();
    fprintf(stderr, "Error: could not import module 'MyModule'\n");
  }

  // Exit
  PyMem_RawFree(program);
  return 0;
}
