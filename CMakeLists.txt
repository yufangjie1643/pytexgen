CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
PROJECT(TexGen LANGUAGES C CXX)

# ============================================================================
# scikit-build-core detection
# When building via `pip install .` or `python -m build`, SKBUILD is set
# automatically by scikit-build-core.
# ============================================================================
if(DEFINED SKBUILD)
    message(STATUS "Building via scikit-build-core (wheel mode)")
    # In wheel mode, find the Python interpreter that scikit-build-core provides
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# ============================================================================
# Compiler flags
# ============================================================================
ADD_COMPILE_DEFINITIONS(SWIG_NO_CRT_SECURE_NO_DEPRECATE)
IF(MSVC)
    ADD_COMPILE_DEFINITIONS(_CRT_SECURE_NO_DEPRECATE)
ENDIF()

IF(UNIX)
    ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>)
    # Ensure all static libs are built with PIC so they can link into shared _Core
    SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

# ============================================================================
# Build options
# ============================================================================
SET(BUILD_RENDERER OFF CACHE BOOL "Enable rendering using VTK.")
SET(BUILD_PYTHON_INTERFACE ON CACHE BOOL "Enable the Python interface.")
SET(BUILD_GUI OFF CACHE BOOL "Build the graphical user interface.")
SET(BUILD_EXAMPLES OFF CACHE BOOL "Build the examples.")
SET(BUILD_UNIT_TESTS OFF CACHE BOOL "Build the unit tests to ensure the software is working as expected.")
SET(BUILD_CASCADE_EXPORT OFF CACHE BOOL "Build module to export geometry to Open CASCADE.")
SET(BUILD_SHARED OFF CACHE BOOL "Build shared libraries (set OFF for wheel packaging)")
SET(BUILD_DOCUMENTATION OFF CACHE BOOL "Build documentation using doxygen")
SET(BUILD_PROFILE OFF CACHE BOOL "Build profiling")
SET(TEXGEN_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP acceleration for C++ point-classification loops")
SET(TEXGEN_ENABLE_NATIVE_OPTIMIZATIONS OFF CACHE BOOL "Enable architecture-native compiler flags such as -march=native")
SET(TEXGEN_REGENERATE_SWIG OFF CACHE BOOL "Regenerate Python SWIG wrappers instead of using committed generated files")

# ============================================================================
# Release-mode optimization flags
# Keep wheel builds portable by default.  Architecture-native flags are useful
# for local benchmarking, but they can make wheels fail on older CPUs.
# ============================================================================
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
ENDIF()

IF(MSVC)
    # Keep wheel builds fast without replacing user/toolchain defaults.
    ADD_COMPILE_OPTIONS(
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Ob2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/fp:fast>
    )
    SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
ELSE()
    ADD_COMPILE_OPTIONS(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fno-finite-math-only>
    )
    IF(TEXGEN_ENABLE_NATIVE_OPTIMIZATIONS)
        INCLUDE(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-march=native" TEXGEN_HAS_MARCH_NATIVE)
        IF(TEXGEN_HAS_MARCH_NATIVE)
            ADD_COMPILE_OPTIONS($<$<CONFIG:Release>:-march=native>)
        ENDIF()
    ENDIF()
ENDIF()

# ============================================================================
# Output paths
# ============================================================================
IF(WIN32)
    SET(REAL_OUTPUT_PATH ${TexGen_BINARY_DIR}/$(OutDir))
ELSE()
    SET(REAL_OUTPUT_PATH ${TexGen_BINARY_DIR})
ENDIF()

# ============================================================================
# OpenMP (optional; Python numpy/torch voxelization is the default acceleration path)
# ============================================================================
IF(TEXGEN_ENABLE_OPENMP)
    FIND_PACKAGE(OpenMP)
    IF(OpenMP_CXX_FOUND)
        MESSAGE(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
    ELSE()
        MESSAGE(WARNING "TEXGEN_ENABLE_OPENMP=ON but OpenMP was not found - build will be single-threaded")
    ENDIF()
ELSE()
    MESSAGE(STATUS "OpenMP disabled (TEXGEN_ENABLE_OPENMP=OFF)")
ENDIF()

IF(BUILD_GUI)
    IF(NOT BUILD_RENDERER)
        MESSAGE("The renderer is needed in order to build the graphical user interface.\nSet BUILD_RENDERER ON.")
    ENDIF()
    IF(NOT BUILD_PYTHON_INTERFACE)
        MESSAGE("The Python interface is needed in order to build the graphical user interface.\nSet BUILD_PYTHON_INTERFACE ON.")
    ENDIF()
ENDIF()

# ============================================================================
# Subdirectories - core C++ libraries (always built as static for bundling)
# ============================================================================
ADD_SUBDIRECTORY(Core)
ADD_SUBDIRECTORY(Triangle)
ADD_SUBDIRECTORY(Octree)
ADD_SUBDIRECTORY(tinyxml)
ADD_SUBDIRECTORY(CSparse)
ADD_SUBDIRECTORY(tetgenlib)
ADD_SUBDIRECTORY(LatinHypercube)

IF(BUILD_RENDERER)
    ADD_SUBDIRECTORY(Renderer)
ENDIF()
IF(BUILD_CASCADE_EXPORT)
    ADD_SUBDIRECTORY(Export)
ENDIF()
IF(BUILD_PYTHON_INTERFACE)
    ADD_SUBDIRECTORY(Python)
ENDIF()
IF(BUILD_GUI)
    ADD_SUBDIRECTORY(GUI)
ENDIF()
IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(Examples)
ENDIF()
IF(BUILD_UNIT_TESTS)
    ADD_SUBDIRECTORY(UnitTests)
ENDIF()
IF(BUILD_PROFILE)
    ADD_SUBDIRECTORY(ProfileTests)
ENDIF()

SET(LIBRARY_OUTPUT_PATH ${TexGen_BINARY_DIR} CACHE INTERNAL "Single output directory for building all libraries.")
SET(EXECUTABLE_OUTPUT_PATH ${TexGen_BINARY_DIR} CACHE INTERNAL "Single output directory for building all executables.")
CONFIGURE_FILE(${TexGen_SOURCE_DIR}/TexGenConfig.cmake.in
               ${TexGen_BINARY_DIR}/TexGenConfig.cmake)
