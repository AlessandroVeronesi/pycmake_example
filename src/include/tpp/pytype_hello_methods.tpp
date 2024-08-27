#ifndef __HELLO_METHODS_TPP__
#define __HELLO_METHODS_TPP__

static PyObject*
Hello_getValue(HelloObject* self, PyObject* Py_UNUSED(ignored))
{
   return Py_BuildValue("i", self->ob_ival);
}

static PyObject*
Hello_setValue(HelloObject* self, PyObject* args, PyObject* kwds)
{
    // Inputs as keywords list
    static char *kwlist[] = {"value", NULL};
    
    // Parse inputs as tuples or keywords list
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &self->ob_ival)) {
        PyErr_SetString(MyModuleError, "Cannot parse input");
        return NULL;
    }

    Py_RETURN_NONE;
}

#endif