#ifndef __PYTYPE_HELLO_H__
#define __PYTYPE_HELLO_H__


//* === Hello Object Definition === *//
typedef struct {
    PyObject_HEAD
    int ob_ival;
} HelloObject;


//* === Hello Class Destructor === *//
static void
Hello_dealloc(HelloObject* self)
{
    //Py_XDECREF(self->member); // FIXME: For future usage
    Py_TYPE(self)->tp_free((PyObject*) self);
}


//* === Hello Class Constructor === *//
static PyObject*
Hello_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    HelloObject* self;
    self = (HelloObject*) type->tp_alloc(type, 0);
    if (self != NULL) {
        //self->member = ... // FIXME: Fot future usage
        //if(self->member == NULL) ...
        self->ob_ival = 0;
    }
    return (PyObject*) self;
}


//* === Hello Class Init === *//
static int
Hello_init(HelloObject* self, PyObject* args, PyObject* kwds)
{
    // Inputs as keywords list
    static char *kwlist[] = {"value", NULL};
    
    // Parse inputs as tuples or keywords list
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &self->ob_ival)) { return -1; }

    // if(pyobjectptr) {  // FIXME: for future usage
    //     Py_XSETREF(self->member, Py_NewRef(pyobjectptr))
    // }

    return 0;
}


//* === Hello Type Members === *//
static PyMemberDef Hello_members[] = {
    //{"member", Py_T_OBJECT_EX, offsetof(HelloObject, member), 0, "member name"}, // FIXME: For future usage
    {"value", Py_T_INT, offsetof(HelloObject, ob_ival), 0, "hello object value"},
    {NULL}
};


//* === Hello Type Methods === *//
static PyObject* Hello_getValue(HelloObject* self, PyObject* Py_UNUSED(ignored));
static PyObject* Hello_setValue(HelloObject* self, PyObject* args, PyObject* kwds);

static PyMethodDef Hello_methods[] = {
    {"get_value", (PyCFunction) Hello_getValue, METH_NOARGS,  "Return the hello's value"},
    {"set_value", (PyCFunction) Hello_setValue, METH_VARARGS | METH_KEYWORDS, "Set the hello's value"},
    {NULL}
};


//* === Hello Type Definition === *//
static PyTypeObject HelloType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MyModule.hello",                   /* tp_name */
    sizeof(HelloObject),                /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor) Hello_dealloc,         /* tp_dealloc */
    0,                                  /* tp_vectorcall_offset */
    0,                                  /* tp_getattr */
    0,                                  /* tp_setattr */
    0,                                  /* tp_as_async */
    0,                                  /* tp_repr */
    0,                                  /* tp_as_number */
    0,                                  /* tp_as_sequence */
    0,                                  /* tp_as_mapping */
    0,                                  /* tp_hash */
    0,                                  /* tp_call */
    0,                                  /* tp_str */
    0,                                  /* tp_getattro */
    0,                                  /* tp_setattro */
    0,                                  /* tp_as_buffer */
    //Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                 /* tp_flags */
    Py_TPFLAGS_DEFAULT,                 /* tp_flags */
    PyDoc_STR("Custom hello object"),   /* tp_doc */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    Hello_methods,                      /* tp_methods */
    Hello_members,                      /* tp_members */
    0,                                  /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc) Hello_init,              /* tp_init */
    0,                                  /* tp_alloc */
    Hello_new,                          /* tp_new */
};

// Methods Implementation
#include "tpp/pytype_hello_methods.tpp"

#endif