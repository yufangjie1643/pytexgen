#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../Core/PrecompiledHeaders.h"
#include "../Core/TexGen.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

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

double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

TexGen::XYZ normalized(TexGen::XYZ value) {
    double length = norm3(value.x, value.y, value.z);
    if (length < 1e-12) {
        length = 1e-12;
    }
    value.x /= length;
    value.y /= length;
    value.z /= length;
    return value;
}

TexGen::XYZ cross(const TexGen::XYZ& a, const TexGen::XYZ& b) {
    return TexGen::XYZ(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

void append_vec3(std::vector<double>& values, const TexGen::XYZ& value) {
    values.push_back(value.x);
    values.push_back(value.y);
    values.push_back(value.z);
}

void append_vec2(std::vector<double>& values, const TexGen::XY& value) {
    values.push_back(value.x);
    values.push_back(value.y);
}

void close_and_orient_section(std::vector<TexGen::XY>& section) {
    if (section.empty()) {
        return;
    }

    const TexGen::XY first = section.front();
    const TexGen::XY last = section.back();
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

std::vector<TexGen::XY> fallback_section(const TexGen::CYarn& yarn) {
    const TexGen::CYarnSection* section = yarn.GetYarnSection();
    if (!section) {
        throw std::runtime_error("yarn has no yarn section");
    }

    TexGen::YARN_POSITION_INFORMATION position;
    position.dSectionPosition = 0.0;
    position.iSection = 0;
    position.SectionLengths = yarn.GetYarnSectionLengths();
    int num_points = yarn.GetNumSectionPoints();
    if (num_points < 3) {
        num_points = 40;
    }
    return section->GetSection(position, num_points, true);
}

void append_translations(const TexGen::CDomain& domain,
                         const TexGen::CYarn& yarn,
                         FlatBundle& bundle) {
    std::vector<TexGen::XYZ> translations = domain.GetTranslations(yarn);
    if (translations.empty()) {
        append_vec3(bundle.translations, TexGen::XYZ(0.0, 0.0, 0.0));
        return;
    }
    for (const TexGen::XYZ& translation : translations) {
        append_vec3(bundle.translations, translation);
    }
}

void append_yarn(const TexGen::CYarn& yarn,
                 const TexGen::CDomain& domain,
                 FlatBundle& bundle) {
    const std::vector<TexGen::CSlaveNode>& slaves =
        yarn.GetSlaveNodes(TexGen::CYarn::SURFACE);
    if (slaves.size() < 2) {
        return;
    }

    std::vector<TexGen::XY> section;
    for (const TexGen::CSlaveNode& node : slaves) {
        append_vec3(bundle.positions, node.GetPosition());
        const TexGen::XYZ tangent = normalized(node.GetTangent());
        const TexGen::XYZ up = normalized(node.GetUp());
        const TexGen::XYZ side = normalized(cross(tangent, up));
        append_vec3(bundle.tangents, tangent);
        append_vec3(bundle.ups, up);
        append_vec3(bundle.sides, side);

        if (section.empty() && node.Get2DSectionPoints().size() >= 3) {
            section = node.Get2DSectionPoints();
        }
    }

    if (section.empty()) {
        section = fallback_section(yarn);
    }
    if (section.size() < 3) {
        throw std::runtime_error("yarn section has fewer than 3 points");
    }
    close_and_orient_section(section);
    for (const TexGen::XY& point : section) {
        append_vec2(bundle.sections, point);
    }

    append_translations(domain, yarn, bundle);
    bundle.node_offsets.push_back(
        static_cast<long long>(bundle.positions.size() / 3));
    bundle.section_offsets.push_back(
        static_cast<long long>(bundle.sections.size() / 2));
    bundle.translation_offsets.push_back(
        static_cast<long long>(bundle.translations.size() / 3));
}

void append_aabb_from_positions(const FlatBundle& bundle, FlatBundle& output) {
    if (bundle.positions.empty()) {
        throw std::runtime_error("cannot compute AABB from an empty textile");
    }
    TexGen::XYZ lo(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    TexGen::XYZ hi(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());

    for (std::size_t i = 0; i + 2 < bundle.positions.size(); i += 3) {
        lo.x = std::min(lo.x, bundle.positions[i]);
        lo.y = std::min(lo.y, bundle.positions[i + 1]);
        lo.z = std::min(lo.z, bundle.positions[i + 2]);
        hi.x = std::max(hi.x, bundle.positions[i]);
        hi.y = std::max(hi.y, bundle.positions[i + 1]);
        hi.z = std::max(hi.z, bundle.positions[i + 2]);
    }
    append_vec3(output.aabb, lo);
    append_vec3(output.aabb, hi);
}

void append_domain_aabb(const TexGen::CDomain& domain, FlatBundle& bundle) {
    const std::vector<TexGen::XYZ>& nodes = domain.GetMesh().GetNodes();
    if (nodes.empty()) {
        append_aabb_from_positions(bundle, bundle);
        return;
    }

    TexGen::XYZ lo(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    TexGen::XYZ hi(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());

    for (const TexGen::XYZ& point : nodes) {
        lo.x = std::min(lo.x, point.x);
        lo.y = std::min(lo.y, point.y);
        lo.z = std::min(lo.z, point.z);
        hi.x = std::max(hi.x, point.x);
        hi.y = std::max(hi.y, point.y);
        hi.z = std::max(hi.z, point.z);
    }
    append_vec3(bundle.aabb, lo);
    append_vec3(bundle.aabb, hi);
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
        const Py_ssize_t nbytes =
            static_cast<Py_ssize_t>(values.size() * sizeof(T));
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
    if (cols <= 0) {
        return flat.release();
    }
    PyRef shape(Py_BuildValue("(nn)", rows, cols));
    if (!shape) {
        return nullptr;
    }
    return PyObject_CallMethod(flat.get(), "reshape", "O", shape.get());
}

bool dict_set_steal(PyObject* dict, const char* key, PyObject* value) {
    if (!value) {
        return false;
    }
    const int status = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return status == 0;
}

PyObject* bundle_to_dict(const FlatBundle& bundle) {
    static_assert(sizeof(long long) == 8, "offset arrays require int64 storage");

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

}  // namespace

extern "C" PyObject* TexGenCore_ExtractSnapshotBundleDirect(TexGen::CTextile* textile) {
    if (!textile) {
        PyErr_SetString(PyExc_TypeError, "expected a non-null CTextile pointer");
        return nullptr;
    }

    try {
        TexGen::CDomain* domain = textile->GetDomain();
        if (!domain) {
            PyErr_SetString(PyExc_ValueError, "textile has no assigned domain");
            return nullptr;
        }

        FlatBundle bundle;
        const int num_yarns = textile->GetNumYarns();
        for (int i = 0; i < num_yarns; ++i) {
            TexGen::CYarn* yarn = textile->GetYarn(i);
            if (yarn) {
                append_yarn(*yarn, *domain, bundle);
            }
        }
        append_domain_aabb(*domain, bundle);
        return bundle_to_dict(bundle);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}
