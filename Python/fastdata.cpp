#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace {

class PyRef {
public:
    explicit PyRef(PyObject* obj = nullptr) : obj_(obj) {}
    ~PyRef() { Py_XDECREF(obj_); }

    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;

    PyObject* get() const { return obj_; }
    PyObject* release() {
        PyObject* obj = obj_;
        obj_ = nullptr;
        return obj;
    }
    explicit operator bool() const { return obj_ != nullptr; }

private:
    PyObject* obj_;
};

PyObject* import_core_module() {
    const char* names[] = {"pytexgen._Core", "TexGen._Core", "_Core"};
    PyObject* last_type = nullptr;
    PyObject* last_value = nullptr;
    PyObject* last_traceback = nullptr;

    for (const char* name : names) {
        PyObject* module = PyImport_ImportModule(name);
        if (module) {
            Py_XDECREF(last_type);
            Py_XDECREF(last_value);
            Py_XDECREF(last_traceback);
            return module;
        }
        Py_XDECREF(last_type);
        Py_XDECREF(last_value);
        Py_XDECREF(last_traceback);
        PyErr_Fetch(&last_type, &last_value, &last_traceback);
    }

    PyErr_Restore(last_type, last_value, last_traceback);
    return nullptr;
}

bool core_has_entry(const char* name) {
    PyRef core(import_core_module());
    if (!core) {
        PyErr_Clear();
        return false;
    }
    return PyObject_HasAttrString(core.get(), name) == 1;
}

PyObject* fastdata_provider_info(PyObject*, PyObject*) {
    PyRef info(PyDict_New());
    PyRef capabilities(PyList_New(0));
    if (!info || !capabilities) {
        return nullptr;
    }

    if (core_has_entry("_fastdata_extract_snapshot_bundle_direct")) {
        PyRef cap(PyUnicode_FromString("extract_from_core_pointer"));
        if (!cap || PyList_Append(capabilities.get(), cap.get()) != 0) {
            return nullptr;
        }
    }
    if (core_has_entry("_fastdata_voxelize_exact_direct")) {
        PyRef cap(PyUnicode_FromString("exact_voxelize_from_core_pointer"));
        if (!cap || PyList_Append(capabilities.get(), cap.get()) != 0) {
            return nullptr;
        }
    }

    PyRef version(PyLong_FromLong(3));
    PyRef backend(PyUnicode_FromString("pytexgen._Core"));
    if (!version || !backend) {
        return nullptr;
    }
    if (PyDict_SetItemString(info.get(), "interface_version", version.get()) != 0
        || PyDict_SetItemString(info.get(), "capabilities", capabilities.get()) != 0
        || PyDict_SetItemString(info.get(), "array_backend", backend.get()) != 0) {
        return nullptr;
    }
    return info.release();
}

PyObject* fastdata_extract_snapshot_bundle(PyObject*, PyObject* target) {
    if (PyUnicode_Check(target)) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "_fastdata extracts from SWIG CTextile proxy objects; registered-name "
            "extraction requires a shared TexGenCore runtime");
        return nullptr;
    }

    PyRef core(import_core_module());
    if (!core) {
        return nullptr;
    }
    PyRef extractor(
        PyObject_GetAttrString(core.get(), "_fastdata_extract_snapshot_bundle_direct"));
    if (!extractor) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "pytexgen._Core lacks _fastdata_extract_snapshot_bundle_direct; rebuild pytexgen");
        return nullptr;
    }
    return PyObject_CallFunctionObjArgs(extractor.get(), target, nullptr);
}

PyObject* fastdata_voxelize_exact(PyObject*, PyObject* args) {
    if (!PyTuple_Check(args) || PyTuple_GET_SIZE(args) != 7) {
        PyErr_SetString(
            PyExc_TypeError,
            "voxelize_exact expects textile, nx, ny, nz, orientation_storage, "
            "workers, tolerance");
        return nullptr;
    }
    PyObject* target = PyTuple_GET_ITEM(args, 0);
    if (PyUnicode_Check(target)) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "exact voxelization requires a SWIG CTextile proxy object");
        return nullptr;
    }

    PyRef core(import_core_module());
    if (!core) {
        return nullptr;
    }
    PyRef voxelizer(
        PyObject_GetAttrString(core.get(), "_fastdata_voxelize_exact_direct"));
    if (!voxelizer) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "pytexgen._Core lacks _fastdata_voxelize_exact_direct; rebuild pytexgen");
        return nullptr;
    }
    return PyObject_CallObject(voxelizer.get(), args);
}

PyMethodDef fastdata_methods[] = {
    {
        "provider_info",
        reinterpret_cast<PyCFunction>(fastdata_provider_info),
        METH_NOARGS,
        const_cast<char*>("Return _fastdata provider capabilities."),
    },
    {
        "extract_snapshot_bundle",
        reinterpret_cast<PyCFunction>(fastdata_extract_snapshot_bundle),
        METH_O,
        const_cast<char*>("Extract a SnapshotBundle mapping using _Core C++ pointers."),
    },
    {
        "voxelize_exact",
        reinterpret_cast<PyCFunction>(fastdata_voxelize_exact),
        METH_VARARGS,
        const_cast<char*>("Voxelize a CTextile with the exact TexGen classifier."),
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef fastdata_module = {
    PyModuleDef_HEAD_INIT,
    "_fastdata",
    "CPython provider facade for zero-copy access to _Core C++ operations.",
    -1,
    fastdata_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__fastdata(void) {
    return PyModule_Create(&fastdata_module);
}
