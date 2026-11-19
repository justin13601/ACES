// A stub so setuptools builds the C library as a Python extension module (aces._native); aces.native loads it
// with ctypes.
#include <Python.h>

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
