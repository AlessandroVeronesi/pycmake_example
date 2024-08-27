#ifndef _C_CONV_CPP_
#define _C_CONV_CPP_



//* ================= *//
// C Method: Tensors Convolution
static PyObject* C_hello(PyObject *self, PyObject *args)
{
    int qBits;

    //* --- PARSE INPUT --- *//

    // Parse Input Tuple (with contiguity check)
    if(!PyArg_ParseTuple(args, "i", &qBits)) {
        PyErr_SetString(MyModuleError, "ERROR[NvdlaEmu.C_conv]: Cannot parse input");
        return NULL;
    }

    std::cout << "Hello World, input is " << qBits << std::endl;

    return NULL;
}

#endif
