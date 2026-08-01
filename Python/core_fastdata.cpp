#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../Core/PrecompiledHeaders.h"
#include "../Core/TexGen.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

bool ensure_numpy_api() {
    static bool ready = false;
    if (ready) {
        return true;
    }
    import_array1(false);
    ready = true;
    return true;
}

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

enum ExactOrientationStorage {
    EXACT_ORIENTATION_NONE = 0,
    EXACT_ORIENTATION_DENSE = 1,
    EXACT_ORIENTATION_SPARSE = 2,
};

struct ExactVoxelData {
    std::vector<int> yarn_id;
    std::vector<long long> voxel_indices;
    std::vector<int> orientation_yarn_ids;
    std::vector<double> orientation1;
    std::vector<double> orientation2;
    std::vector<double> aabb;
    int workers = 1;
};

class AllowThreads {
public:
    AllowThreads() : state_(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state_); }

    AllowThreads(const AllowThreads&) = delete;
    AllowThreads& operator=(const AllowThreads&) = delete;

private:
    PyThreadState* state_;
};

struct ExactLayerOrientationData {
    std::vector<long long> voxel_indices;
    std::vector<int> yarn_ids;
    std::vector<double> orientation1;
    std::vector<double> orientation2;
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
                            int numpy_type,
                            Py_ssize_t rows,
                            Py_ssize_t cols) {
    npy_intp dims[2] = {static_cast<npy_intp>(rows), static_cast<npy_intp>(cols)};
    int ndim = 2;
    if (cols <= 0) {
        dims[0] = static_cast<npy_intp>(values.size());
        ndim = 1;
    }
    PyObject* array = PyArray_SimpleNew(ndim, dims, numpy_type);
    if (!array) {
        return nullptr;
    }
    if (!values.empty()) {
        std::memcpy(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            values.data(),
            values.size() * sizeof(T));
    }
    return array;
}

bool dict_set_steal(PyObject* dict, const char* key, PyObject* value) {
    if (!value) {
        return false;
    }
    const int status = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return status == 0;
}

bool dict_set_long(PyObject* dict, const char* key, long value) {
    return dict_set_steal(dict, key, PyLong_FromLong(value));
}

PyObject* bundle_to_dict(const FlatBundle& bundle) {
    static_assert(sizeof(long long) == 8, "offset arrays require int64 storage");

    PyRef dict(PyDict_New());
    if (!dict) {
        return nullptr;
    }

    if (!dict_set_steal(dict.get(), "positions",
                        array_from_vector(bundle.positions, NPY_DOUBLE,
                                          static_cast<Py_ssize_t>(bundle.positions.size() / 3), 3))
        || !dict_set_steal(dict.get(), "tangents",
                           array_from_vector(bundle.tangents, NPY_DOUBLE,
                                             static_cast<Py_ssize_t>(bundle.tangents.size() / 3), 3))
        || !dict_set_steal(dict.get(), "ups",
                           array_from_vector(bundle.ups, NPY_DOUBLE,
                                             static_cast<Py_ssize_t>(bundle.ups.size() / 3), 3))
        || !dict_set_steal(dict.get(), "sides",
                           array_from_vector(bundle.sides, NPY_DOUBLE,
                                             static_cast<Py_ssize_t>(bundle.sides.size() / 3), 3))
        || !dict_set_steal(dict.get(), "node_offsets",
                           array_from_vector(bundle.node_offsets, NPY_INT64,
                                             static_cast<Py_ssize_t>(bundle.node_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "sections",
                           array_from_vector(bundle.sections, NPY_DOUBLE,
                                             static_cast<Py_ssize_t>(bundle.sections.size() / 2), 2))
        || !dict_set_steal(dict.get(), "section_offsets",
                           array_from_vector(bundle.section_offsets, NPY_INT64,
                                             static_cast<Py_ssize_t>(bundle.section_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "translations",
                           array_from_vector(bundle.translations, NPY_DOUBLE,
                                             static_cast<Py_ssize_t>(bundle.translations.size() / 3), 3))
        || !dict_set_steal(dict.get(), "translation_offsets",
                           array_from_vector(bundle.translation_offsets, NPY_INT64,
                                             static_cast<Py_ssize_t>(bundle.translation_offsets.size()), 0))
        || !dict_set_steal(dict.get(), "aabb",
                           array_from_vector(bundle.aabb, NPY_DOUBLE, 2, 3))) {
        return nullptr;
    }
    return dict.release();
}

void validate_exact_voxel_args(int nx,
                               int ny,
                               int nz,
                               int orientation_storage,
                               int workers,
                               double tolerance) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("voxel dimensions must be positive");
    }
    if (orientation_storage < EXACT_ORIENTATION_NONE
        || orientation_storage > EXACT_ORIENTATION_SPARSE) {
        throw std::invalid_argument(
            "orientation storage must be 0 (none), 1 (dense), or 2 (sparse)");
    }
    if (workers < 0) {
        throw std::invalid_argument("workers must be non-negative");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("tolerance must be finite and non-negative");
    }

    const std::size_t x = static_cast<std::size_t>(nx);
    const std::size_t y = static_cast<std::size_t>(ny);
    const std::size_t z = static_cast<std::size_t>(nz);
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (x > max_size / y || x * y > max_size / z) {
        throw std::overflow_error("voxel dimensions overflow addressable memory");
    }
    const std::size_t total = x * y * z;
    if (total > static_cast<std::size_t>(
                    std::numeric_limits<npy_intp>::max())
        || total > static_cast<std::size_t>(
                       std::numeric_limits<long long>::max())) {
        throw std::overflow_error("voxel count exceeds supported array indexing");
    }
    if (orientation_storage != EXACT_ORIENTATION_NONE
        && total > max_size / 3) {
        throw std::overflow_error("orientation array size overflows addressable memory");
    }
}

ExactVoxelData voxelize_exact(TexGen::CTextile& textile,
                              int nx,
                              int ny,
                              int nz,
                              int orientation_storage,
                              int workers,
                              double tolerance) {
    validate_exact_voxel_args(
        nx, ny, nz, orientation_storage, workers, tolerance);

    TexGen::CDomain* domain = textile.GetDomain();
    if (!domain) {
        throw std::invalid_argument("textile has no assigned domain");
    }

    const std::pair<TexGen::XYZ, TexGen::XYZ> bounds =
        domain->GetMesh().GetAABB();
    const TexGen::XYZ lo = bounds.first;
    const TexGen::XYZ hi = bounds.second;
    const TexGen::XYZ spacing(
        (hi.x - lo.x) / static_cast<double>(nx),
        (hi.y - lo.y) / static_cast<double>(ny),
        (hi.z - lo.z) / static_cast<double>(nz));
    if (!(spacing.x > 0.0) || !(spacing.y > 0.0) || !(spacing.z > 0.0)
        || !std::isfinite(spacing.x) || !std::isfinite(spacing.y)
        || !std::isfinite(spacing.z)) {
        throw std::invalid_argument("textile domain must have finite positive extents");
    }

    const std::size_t layer_size =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const std::size_t total = layer_size * static_cast<std::size_t>(nz);

    ExactVoxelData output;
    output.yarn_id.assign(total, -1);
    output.aabb = {lo.x, lo.y, lo.z, hi.x, hi.y, hi.z};
    if (orientation_storage == EXACT_ORIENTATION_DENSE) {
        output.orientation1.assign(total * 3, 0.0);
        output.orientation2.assign(total * 3, 0.0);
    }

    unsigned int default_workers = std::thread::hardware_concurrency();
    if (default_workers == 0) {
        default_workers = 1;
    }
    const unsigned int automatic_worker_limit = 8;
    const int configured_workers = workers > 0
        ? workers
        : static_cast<int>(std::min<unsigned int>(
              std::min(default_workers, automatic_worker_limit),
              static_cast<unsigned int>(std::numeric_limits<int>::max())));
    const int actual_workers = std::max(1, std::min(configured_workers, nz));
    output.workers = actual_workers;

    // Complete lazy geometry construction once. Copies retain the exact
    // double-precision caches but own their slave-node mesh allocations.
    std::vector<TexGen::XYZ> warmup_points(
        1, TexGen::XYZ(
            lo.x + 0.5 * spacing.x,
            lo.y + 0.5 * spacing.y,
            lo.z + 0.5 * spacing.z));
    std::vector<TexGen::POINT_INFO> warmup_information;
    textile.GetPointInformation(warmup_points, warmup_information, tolerance);

    // TexGen caches section/interpolation data inside nominally const point
    // queries. Never share a CTextile across workers: each worker owns an
    // exact in-memory copy with independent mutable caches. Worker zero uses
    // the caller's textile; the remaining workers use virtual deep copies.
    std::vector<std::unique_ptr<TexGen::CTextile> > worker_textiles;
    worker_textiles.reserve(static_cast<std::size_t>(actual_workers - 1));
    for (int index = 1; index < actual_workers; ++index) {
        std::unique_ptr<TexGen::CTextile> copy(textile.Copy());
        if (!copy) {
            throw std::runtime_error("TexGen failed to copy textile for worker");
        }
        worker_textiles.push_back(std::move(copy));
    }

    std::vector<ExactLayerOrientationData> sparse_layers;
    if (orientation_storage == EXACT_ORIENTATION_SPARSE) {
        sparse_layers.resize(static_cast<std::size_t>(nz));
    }

    std::atomic<int> next_layer(0);
    std::atomic<bool> stop(false);
    std::mutex failure_mutex;
    std::exception_ptr failure;

    const auto process_layer = [&](TexGen::CTextile& worker_textile,
                                   int z,
                                   std::vector<TexGen::XYZ>& centers,
                                   std::vector<TexGen::POINT_INFO>& information) {
        const double center_z =
            lo.z + spacing.z * static_cast<double>(z) + 0.5 * spacing.z;
        for (int y = 0; y < ny; ++y) {
            const double center_y =
                lo.y + spacing.y * static_cast<double>(y) + 0.5 * spacing.y;
            for (int x = 0; x < nx; ++x) {
                const double center_x =
                    lo.x + spacing.x * static_cast<double>(x) + 0.5 * spacing.x;
                centers[static_cast<std::size_t>(x)
                        + static_cast<std::size_t>(y) * static_cast<std::size_t>(nx)] =
                    TexGen::XYZ(center_x, center_y, center_z);
            }
        }

        information.clear();
        worker_textile.GetPointInformation(centers, information, tolerance);
        if (information.size() != layer_size) {
            throw std::runtime_error(
                "TexGen GetPointInformation returned an unexpected result size");
        }

        const std::size_t layer_start = static_cast<std::size_t>(z) * layer_size;
        ExactLayerOrientationData* sparse =
            orientation_storage == EXACT_ORIENTATION_SPARSE
            ? &sparse_layers[static_cast<std::size_t>(z)]
            : nullptr;
        for (std::size_t local = 0; local < layer_size; ++local) {
            const TexGen::POINT_INFO& info = information[local];
            const std::size_t flat = layer_start + local;
            output.yarn_id[flat] = info.iYarnIndex;
            if (info.iYarnIndex < 0 || orientation_storage == EXACT_ORIENTATION_NONE) {
                continue;
            }

            if (orientation_storage == EXACT_ORIENTATION_DENSE) {
                const std::size_t base = flat * 3;
                const TexGen::XYZ secondary = normalized(
                    cross(info.Orientation, info.Up));
                output.orientation1[base] = info.Orientation.x;
                output.orientation1[base + 1] = info.Orientation.y;
                output.orientation1[base + 2] = info.Orientation.z;
                output.orientation2[base] = secondary.x;
                output.orientation2[base + 1] = secondary.y;
                output.orientation2[base + 2] = secondary.z;
            } else {
                sparse->voxel_indices.push_back(static_cast<long long>(flat));
                sparse->yarn_ids.push_back(info.iYarnIndex);
                append_vec3(sparse->orientation1, info.Orientation);
                append_vec3(
                    sparse->orientation2,
                    normalized(cross(info.Orientation, info.Up)));
            }
        }
    };

    const auto worker_loop = [&](int worker_index) {
        try {
            TexGen::CTextile& worker_textile = worker_index == 0
                ? textile
                : *worker_textiles[static_cast<std::size_t>(worker_index - 1)];
            std::vector<TexGen::XYZ> centers(layer_size);
            std::vector<TexGen::POINT_INFO> information;
            while (!stop.load(std::memory_order_relaxed)) {
                const int z = next_layer.fetch_add(1, std::memory_order_relaxed);
                if (z >= nz) {
                    break;
                }
                process_layer(worker_textile, z, centers, information);
            }
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) {
                failure = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(actual_workers - 1));
    try {
        for (int index = 1; index < actual_workers; ++index) {
            threads.emplace_back(worker_loop, index);
        }
    } catch (...) {
        stop.store(true, std::memory_order_relaxed);
        for (std::thread& thread : threads) {
            thread.join();
        }
        throw;
    }
    worker_loop(0);
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }

    if (orientation_storage == EXACT_ORIENTATION_SPARSE) {
        std::size_t occupied = 0;
        for (const ExactLayerOrientationData& layer : sparse_layers) {
            occupied += layer.voxel_indices.size();
        }
        output.voxel_indices.reserve(occupied);
        output.orientation_yarn_ids.reserve(occupied);
        output.orientation1.reserve(occupied * 3);
        output.orientation2.reserve(occupied * 3);
        for (ExactLayerOrientationData& layer : sparse_layers) {
            output.voxel_indices.insert(
                output.voxel_indices.end(),
                layer.voxel_indices.begin(),
                layer.voxel_indices.end());
            output.orientation_yarn_ids.insert(
                output.orientation_yarn_ids.end(),
                layer.yarn_ids.begin(),
                layer.yarn_ids.end());
            output.orientation1.insert(
                output.orientation1.end(),
                layer.orientation1.begin(),
                layer.orientation1.end());
            output.orientation2.insert(
                output.orientation2.end(),
                layer.orientation2.begin(),
                layer.orientation2.end());
        }
    }
    return output;
}

PyObject* exact_voxel_data_to_dict(const ExactVoxelData& data,
                                   int nx,
                                   int ny,
                                   int nz,
                                   int orientation_storage) {
    static_assert(sizeof(int) == 4, "exact yarn ids require 32-bit int storage");
    static_assert(sizeof(long long) == 8, "exact voxel indices require int64 storage");

    PyRef dict(PyDict_New());
    if (!dict) {
        return nullptr;
    }
    const Py_ssize_t total = static_cast<Py_ssize_t>(data.yarn_id.size());
    if (!dict_set_steal(
            dict.get(), "yarn_id",
            array_from_vector(data.yarn_id, NPY_INT32, total, 0))
        || !dict_set_steal(
            dict.get(), "aabb",
            array_from_vector(data.aabb, NPY_DOUBLE, 2, 3))
        || !dict_set_long(dict.get(), "nx", nx)
        || !dict_set_long(dict.get(), "ny", ny)
        || !dict_set_long(dict.get(), "nz", nz)) {
        return nullptr;
    }

    if (orientation_storage == EXACT_ORIENTATION_DENSE) {
        if (!dict_set_steal(
                dict.get(), "orientation1",
                array_from_vector(data.orientation1, NPY_DOUBLE, total, 3))
            || !dict_set_steal(
                dict.get(), "orientation2",
                array_from_vector(data.orientation2, NPY_DOUBLE, total, 3))) {
            return nullptr;
        }
    } else if (orientation_storage == EXACT_ORIENTATION_SPARSE) {
        const Py_ssize_t occupied =
            static_cast<Py_ssize_t>(data.voxel_indices.size());
        if (!dict_set_steal(
                dict.get(), "voxel_indices",
                array_from_vector(data.voxel_indices, NPY_INT64, occupied, 0))
            || !dict_set_steal(
                dict.get(), "orientation_yarn_ids",
                array_from_vector(
                    data.orientation_yarn_ids, NPY_INT32, occupied, 0))
            || !dict_set_steal(
                dict.get(), "orientation1",
                array_from_vector(data.orientation1, NPY_DOUBLE, occupied, 3))
            || !dict_set_steal(
                dict.get(), "orientation2",
                array_from_vector(data.orientation2, NPY_DOUBLE, occupied, 3))) {
            return nullptr;
        }
    }

    if (!dict_set_long(dict.get(), "workers", data.workers)) {
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
    if (!ensure_numpy_api()) {
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

extern "C" PyObject* TexGenCore_VoxelizeExactDirect(
    TexGen::CTextile* textile,
    int nx,
    int ny,
    int nz,
    int orientation_storage,
    int workers,
    double tolerance) {
    if (!textile) {
        PyErr_SetString(PyExc_TypeError, "expected a non-null CTextile pointer");
        return nullptr;
    }
    if (!ensure_numpy_api()) {
        return nullptr;
    }

    try {
        ExactVoxelData data;
        {
            AllowThreads allow_threads;
            data = voxelize_exact(
                *textile, nx, ny, nz, orientation_storage, workers, tolerance);
        }
        return exact_voxel_data_to_dict(
            data, nx, ny, nz, orientation_storage);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}
