#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class PyRef {
public:
    explicit PyRef(PyObject* obj = nullptr) : obj_(obj) {}
    ~PyRef() { Py_XDECREF(obj_); }

    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;

    PyObject* get() const { return obj_; }
    explicit operator bool() const { return obj_ != nullptr; }

    PyObject* release() {
        PyObject* out = obj_;
        obj_ = nullptr;
        return out;
    }

private:
    PyObject* obj_;
};

struct XY2 {
    double x;
    double y;
};

struct FlatBundle {
    std::vector<double> positions;
    std::vector<double> tangents;
    std::vector<double> ups;
    std::vector<double> sides;
    std::vector<long long> node_offsets{0};
    std::vector<double> sections;
    std::vector<long long> section_offsets{0};
    std::vector<double> translations;
    std::vector<long long> translation_offsets{0};
    std::vector<double> aabb;
};

PyObject* call_noargs(PyObject* obj, const char* name) {
    PyRef method(PyObject_GetAttrString(obj, name));
    if (!method) {
        return nullptr;
    }
    return PyObject_CallNoArgs(method.get());
}

bool read_double_attr(PyObject* obj, const char* name, double& out) {
    PyRef attr(PyObject_GetAttrString(obj, name));
    if (!attr) {
        return false;
    }
    out = PyFloat_AsDouble(attr.get());
    return !PyErr_Occurred();
}

bool read_xyz(PyObject* obj, double& x, double& y, double& z) {
    return read_double_attr(obj, "x", x)
        && read_double_attr(obj, "y", y)
        && read_double_attr(obj, "z", z);
}

bool read_xy(PyObject* obj, double& x, double& y) {
    return read_double_attr(obj, "x", x)
        && read_double_attr(obj, "y", y);
}

void normalize(double& x, double& y, double& z) {
    double n = std::sqrt(x * x + y * y + z * z);
    if (n < 1e-12) {
        n = 1e-12;
    }
    x /= n;
    y /= n;
    z /= n;
}

void append_vec3(std::vector<double>& values, double x, double y, double z) {
    values.push_back(x);
    values.push_back(y);
    values.push_back(z);
}

void append_vec2(std::vector<double>& values, const XY2& p) {
    values.push_back(p.x);
    values.push_back(p.y);
}

bool collect_translations(PyObject* domain, PyObject* yarn, std::vector<double>& out) {
    PyRef translations(PyObject_CallMethod(domain, "GetTranslations", "O", yarn));
    if (!translations) {
        PyErr_Clear();
        append_vec3(out, 0.0, 0.0, 0.0);
        return true;
    }

    PyRef seq(PySequence_Fast(
        translations.get(), "CDomain.GetTranslations returned a non-sequence"));
    if (!seq) {
        return false;
    }

    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    if (n == 0) {
        append_vec3(out, 0.0, 0.0, 0.0);
        return true;
    }

    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < n; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (!read_xyz(items[i], x, y, z)) {
            return false;
        }
        append_vec3(out, x, y, z);
    }
    return true;
}

bool section_from_node(PyObject* node, std::vector<XY2>& section) {
    PyRef points(call_noargs(node, "Get2DSectionPoints"));
    if (!points) {
        PyErr_Clear();
        return true;
    }

    PyRef seq(PySequence_Fast(
        points.get(), "CSlaveNode.Get2DSectionPoints returned a non-sequence"));
    if (!seq) {
        PyErr_Clear();
        return true;
    }

    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    if (n < 3) {
        return true;
    }

    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    section.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        XY2 p{0.0, 0.0};
        if (!read_xy(items[i], p.x, p.y)) {
            return false;
        }
        section.push_back(p);
    }
    return true;
}

bool section_from_yarn_fallback(PyObject* yarn, std::vector<XY2>& section) {
    PyRef yarn_section(call_noargs(yarn, "GetYarnSection"));
    if (!yarn_section) {
        return false;
    }
    PyRef section_obj(PyObject_CallMethod(yarn_section.get(), "GetSection", "d", 0.0));
    if (!section_obj) {
        return false;
    }
    PyRef points(PyObject_CallMethod(section_obj.get(), "GetPoints", "iO", 40, Py_True));
    if (!points) {
        return false;
    }
    PyRef seq(PySequence_Fast(points.get(), "CSection.GetPoints returned a non-sequence"));
    if (!seq) {
        return false;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
    if (n < 3) {
        PyErr_SetString(PyExc_ValueError, "yarn section has fewer than 3 points");
        return false;
    }
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    section.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        XY2 p{0.0, 0.0};
        if (!read_xy(items[i], p.x, p.y)) {
            return false;
        }
        section.push_back(p);
    }
    return true;
}

void close_and_orient_section(std::vector<XY2>& section) {
    if (section.empty()) {
        return;
    }

    const XY2& first = section.front();
    const XY2& last = section.back();
    if (std::abs(first.x - last.x) > 1e-12 || std::abs(first.y - last.y) > 1e-12) {
        section.push_back(first);
    }

    double area2 = 0.0;
    for (std::size_t i = 0; i + 1 < section.size(); ++i) {
        area2 += section[i].x * section[i + 1].y - section[i + 1].x * section[i].y;
    }
    if (area2 < 0.0) {
        std::reverse(section.begin(), section.end());
    }
}

bool append_yarn(PyObject* yarn, PyObject* domain, FlatBundle& bundle) {
    PyRef build_method(PyObject_GetAttrString(yarn, "BuildYarnIfNeeded"));
    if (build_method) {
        PyRef built(PyObject_CallFunction(build_method.get(), "i", 1 | 2 | 4));
        if (!built) {
            return false;
        }
    } else {
        PyErr_Clear();
    }

    PyRef slaves(PyObject_CallMethod(yarn, "GetSlaveNodes", "i", 2));
    if (!slaves) {
        return false;
    }
    PyRef slave_seq(PySequence_Fast(slaves.get(), "CYarn.GetSlaveNodes returned a non-sequence"));
    if (!slave_seq) {
        return false;
    }

    const Py_ssize_t n_slaves = PySequence_Fast_GET_SIZE(slave_seq.get());
    if (n_slaves < 2) {
        return true;
    }

    std::vector<XY2> section;
    PyObject** slave_items = PySequence_Fast_ITEMS(slave_seq.get());
    for (Py_ssize_t i = 0; i < n_slaves; ++i) {
        PyObject* node = slave_items[i];

        PyRef position(call_noargs(node, "GetPosition"));
        PyRef tangent(call_noargs(node, "GetTangent"));
        PyRef up(call_noargs(node, "GetUp"));
        if (!position || !tangent || !up) {
            return false;
        }

        double px = 0.0, py = 0.0, pz = 0.0;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        double ux = 0.0, uy = 0.0, uz = 0.0;
        if (!read_xyz(position.get(), px, py, pz)
            || !read_xyz(tangent.get(), tx, ty, tz)
            || !read_xyz(up.get(), ux, uy, uz)) {
            return false;
        }

        normalize(tx, ty, tz);
        normalize(ux, uy, uz);
        double sx = ty * uz - tz * uy;
        double sy = tz * ux - tx * uz;
        double sz = tx * uy - ty * ux;
        normalize(sx, sy, sz);

        append_vec3(bundle.positions, px, py, pz);
        append_vec3(bundle.tangents, tx, ty, tz);
        append_vec3(bundle.ups, ux, uy, uz);
        append_vec3(bundle.sides, sx, sy, sz);

        if (section.empty() && !section_from_node(node, section)) {
            return false;
        }
    }

    if (section.empty() && !section_from_yarn_fallback(yarn, section)) {
        return false;
    }
    close_and_orient_section(section);
    for (const XY2& p : section) {
        append_vec2(bundle.sections, p);
    }

    if (!collect_translations(domain, yarn, bundle.translations)) {
        return false;
    }

    bundle.node_offsets.push_back(
        static_cast<long long>(bundle.positions.size() / 3));
    bundle.section_offsets.push_back(
        static_cast<long long>(bundle.sections.size() / 2));
    bundle.translation_offsets.push_back(
        static_cast<long long>(bundle.translations.size() / 3));
    return true;
}

bool compute_aabb_from_positions(const FlatBundle& bundle, std::vector<double>& out) {
    if (bundle.positions.empty()) {
        PyErr_SetString(PyExc_ValueError, "cannot compute AABB from an empty textile");
        return false;
    }

    double lo[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    double hi[3] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (std::size_t i = 0; i + 2 < bundle.positions.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = bundle.positions[i + static_cast<std::size_t>(axis)];
            lo[axis] = std::min(lo[axis], value);
            hi[axis] = std::max(hi[axis], value);
        }
    }
    out.assign({lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]});
    return true;
}

bool append_domain_aabb(PyObject* domain, const FlatBundle& bundle, std::vector<double>& out) {
    PyRef mesh(call_noargs(domain, "GetMesh"));
    if (!mesh) {
        return false;
    }
    PyRef nodes(call_noargs(mesh.get(), "GetNodes"));
    if (!nodes) {
        return false;
    }
    PyRef node_seq(PySequence_Fast(nodes.get(), "domain mesh nodes are not a sequence"));
    if (!node_seq) {
        return false;
    }

    const Py_ssize_t n_nodes = PySequence_Fast_GET_SIZE(node_seq.get());
    if (n_nodes == 0) {
        return compute_aabb_from_positions(bundle, out);
    }

    double lo[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    double hi[3] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    PyObject** items = PySequence_Fast_ITEMS(node_seq.get());
    for (Py_ssize_t i = 0; i < n_nodes; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (!read_xyz(items[i], x, y, z)) {
            return false;
        }
        lo[0] = std::min(lo[0], x);
        lo[1] = std::min(lo[1], y);
        lo[2] = std::min(lo[2], z);
        hi[0] = std::max(hi[0], x);
        hi[1] = std::max(hi[1], y);
        hi[2] = std::max(hi[2], z);
    }
    out.assign({lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]});
    return true;
}

template <typename T>
PyObject* array_from_vector(const std::vector<T>& values,
                            const char* array_typecode,
                            const char* numpy_dtype,
                            Py_ssize_t rows,
                            Py_ssize_t cols) {
    PyRef array_mod(PyImport_ImportModule("array"));
    if (!array_mod) {
        return nullptr;
    }
    PyRef array_cls(PyObject_GetAttrString(array_mod.get(), "array"));
    PyRef typecode(PyUnicode_FromString(array_typecode));
    if (!array_cls || !typecode) {
        return nullptr;
    }
    PyRef storage(PyObject_CallFunctionObjArgs(array_cls.get(), typecode.get(), nullptr));
    if (!storage) {
        return nullptr;
    }
    if (!values.empty()) {
        const char* bytes = reinterpret_cast<const char*>(values.data());
        const Py_ssize_t nbytes = static_cast<Py_ssize_t>(values.size() * sizeof(T));
        PyRef loaded(PyObject_CallMethod(storage.get(), "frombytes", "y#", bytes, nbytes));
        if (!loaded) {
            return nullptr;
        }
    }

    PyRef numpy(PyImport_ImportModule("numpy"));
    PyRef dtype(PyUnicode_FromString(numpy_dtype));
    if (!numpy || !dtype) {
        return nullptr;
    }
    PyRef flat(PyObject_CallMethod(numpy.get(), "frombuffer", "OO", storage.get(), dtype.get()));
    if (!flat) {
        return nullptr;
    }
    if (cols > 0) {
        PyRef shape(Py_BuildValue("(nn)", rows, cols));
        if (!shape) {
            return nullptr;
        }
        return PyObject_CallMethod(flat.get(), "reshape", "O", shape.get());
    }
    return flat.release();
}

bool dict_set_steal(PyObject* dict, const char* key, PyObject* value) {
    if (!value) {
        return false;
    }
    const int ok = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return ok == 0;
}

PyObject* bundle_to_dict(const FlatBundle& bundle) {
    static_assert(sizeof(long long) == 8, "offset arrays require 64-bit long long");

    PyRef dict(PyDict_New());
    if (!dict) {
        return nullptr;
    }

    if (!dict_set_steal(dict.get(), "positions",
                        array_from_vector(bundle.positions, "d", "float64",
                                          static_cast<Py_ssize_t>(bundle.positions.size() / 3), 3))
        || !dict_set_steal(dict.get(), "tangents",
                           array_from_vector(bundle.tangents, "d", "float64",
                                             static_cast<Py_ssize_t>(bundle.tangents.size() / 3), 3))
        || !dict_set_steal(dict.get(), "ups",
                           array_from_vector(bundle.ups, "d", "float64",
                                             static_cast<Py_ssize_t>(bundle.ups.size() / 3), 3))
        || !dict_set_steal(dict.get(), "sides",
                           array_from_vector(bundle.sides, "d", "float64",
                                             static_cast<Py_ssize_t>(bundle.sides.size() / 3), 3))
        || !dict_set_steal(dict.get(), "node_offsets",
                           array_from_vector(bundle.node_offsets, "q", "int64",
                                             static_cast<Py_ssize_t>(bundle.node_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "sections",
                           array_from_vector(bundle.sections, "d", "float64",
                                             static_cast<Py_ssize_t>(bundle.sections.size() / 2), 2))
        || !dict_set_steal(dict.get(), "section_offsets",
                           array_from_vector(bundle.section_offsets, "q", "int64",
                                             static_cast<Py_ssize_t>(bundle.section_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "translations",
                           array_from_vector(bundle.translations, "d", "float64",
                                             static_cast<Py_ssize_t>(bundle.translations.size() / 3), 3))
        || !dict_set_steal(dict.get(), "translation_offsets",
                           array_from_vector(bundle.translation_offsets, "q", "int64",
                                             static_cast<Py_ssize_t>(bundle.translation_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "aabb",
                           array_from_vector(bundle.aabb, "d", "float64", 2, 3))) {
        return nullptr;
    }

    return dict.release();
}

PyObject* fastdata_provider_info(PyObject*, PyObject*) {
    PyRef info(PyDict_New());
    PyRef capabilities(PyList_New(0));
    if (!info || !capabilities) {
        return nullptr;
    }
    PyRef cap(PyUnicode_FromString("extract_from_swig_proxy"));
    PyRef version(PyLong_FromLong(1));
    PyRef backend(PyUnicode_FromString("numpy.frombuffer"));
    if (!cap || !version || !backend) {
        return nullptr;
    }
    if (PyList_Append(capabilities.get(), cap.get()) != 0) {
        return nullptr;
    }
    if (PyDict_SetItemString(info.get(), "interface_version", version.get()) != 0) {
        return nullptr;
    }
    if (PyDict_SetItemString(info.get(), "capabilities", capabilities.get()) != 0) {
        return nullptr;
    }
    if (PyDict_SetItemString(info.get(), "array_backend", backend.get()) != 0) {
        return nullptr;
    }
    return info.release();
}

PyObject* fastdata_extract_snapshot_bundle(PyObject*, PyObject* target) {
    if (PyUnicode_Check(target)) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "_fastdata currently extracts from SWIG CTextile proxy objects; "
            "registered-name extraction requires a shared TexGenCore runtime");
        return nullptr;
    }

    PyRef domain(call_noargs(target, "GetDomain"));
    if (!domain) {
        PyErr_SetString(
            PyExc_TypeError,
            "extract_snapshot_bundle expects a SWIG CTextile-like object with GetDomain()");
        return nullptr;
    }

    PyRef num_obj(call_noargs(target, "GetNumYarns"));
    if (!num_obj) {
        return nullptr;
    }
    const long num_yarns = PyLong_AsLong(num_obj.get());
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (num_yarns < 0) {
        PyErr_SetString(PyExc_ValueError, "textile returned a negative yarn count");
        return nullptr;
    }

    FlatBundle bundle;
    for (long i = 0; i < num_yarns; ++i) {
        PyRef yarn(PyObject_CallMethod(target, "GetYarn", "l", i));
        if (!yarn) {
            return nullptr;
        }
        if (!append_yarn(yarn.get(), domain.get(), bundle)) {
            return nullptr;
        }
    }

    if (!append_domain_aabb(domain.get(), bundle, bundle.aabb)) {
        return nullptr;
    }
    return bundle_to_dict(bundle);
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
        const_cast<char*>("Extract a SnapshotBundle mapping from a SWIG CTextile proxy."),
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef fastdata_module = {
    PyModuleDef_HEAD_INIT,
    "_fastdata",
    "CPython accelerator for pytexgen SnapshotBundle extraction.",
    -1,
    fastdata_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__fastdata(void) {
    return PyModule_Create(&fastdata_module);
}
