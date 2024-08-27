#ifndef _C_METHODS_H_
#define _C_METHODS_H_

//* === Error Object  === *//
static PyObject *MyModuleError;

//* ====  C Methods  ==== *//
//static PyObject* C_hello(PyObject *self, PyObject *args);
static PyObject* C_hello(PyObject *self, PyObject *args);

//* ====  TPP  ==== *//
#include "tpp/c_hello.tpp"

#endif
