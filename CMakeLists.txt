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
ADD_DEFINITIONS(-DSWIG_NO_CRT_SECURE_NO_DEPRECATE)
IF(MSVC)
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
ENDIF()

IF(UNIX)
    ADD_DEFINITIONS(-Wno-deprecated)
    # Ensure all static libs are built with PIC so they can link into shared _Core
    SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

# ============================================================================
# OpenMP (for parallelizing hot loops such as CTextile::GetPointInformation)
# ============================================================================
FIND_PACKAGE(OpenMP)
IF(OpenMP_CXX_FOUND)
    MESSAGE(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
ELSE()
    MESSAGE(WARNING "OpenMP not found - build will be single-threaded")
ENDIF()

# ============================================================================
# Release-mode optimization flags
# Force -O3 / /O2 + architecture-native vectorization when building Release.
# Users can still override via -DCMAKE_CXX_FLAGS_RELEASE=... on the command line.
# ============================================================================
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
ENDIF()

IF(MSVC)
    # /O2 is MSVC's max speed; /fp:fast relaxes FP for vectorization; /GL enables LTCG
    SET(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /GL /fp:fast /DNDEBUG"
        CACHE STRING "Release C++ flags" FORCE)
    SET(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /GL /fp:fast /DNDEBUG"
        CACHE STRING "Release C flags" FORCE)
    SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG" CACHE STRING "" FORCE)
    SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG" CACHE STRING "" FORCE)
    SET(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG" CACHE STRING "" FORCE)
ELSE()
    # -O3 + native arch (skip -march=native on aarch64 CI where it may be unsupported)
    INCLUDE(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" TEXGEN_HAS_MARCH_NATIVE)
    SET(_TEXGEN_OPT_FLAGS "-O3 -DNDEBUG -ffast-math -fno-finite-math-only")
    IF(TEXGEN_HAS_MARCH_NATIVE)
        SET(_TEXGEN_OPT_FLAGS "${_TEXGEN_OPT_FLAGS} -march=native")
    ENDIF()
    SET(CMAKE_CXX_FLAGS_RELEASE "${_TEXGEN_OPT_FLAGS}"
        CACHE STRING "Release C++ flags" FORCE)
    SET(CMAKE_C_FLAGS_RELEASE "${_TEXGEN_OPT_FLAGS}"
        CACHE STRING "Release C flags" FORCE)
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
# Build options
# ============================================================================
SET(BUILD_RENDERER ON CACHE BOOL "Enable rendering using VTK.")
SET(BUILD_PYTHON_INTERFACE ON CACHE BOOL "Enable the Python interface.")
SET(BUILD_GUI ON CACHE BOOL "Build the graphical user interface.")
SET(BUILD_EXAMPLES OFF CACHE BOOL "Build the examples.")
SET(BUILD_UNIT_TESTS OFF CACHE BOOL "Build the unit tests to ensure the software is working as expected.")
SET(BUILD_CASCADE_EXPORT OFF CACHE BOOL "Build module to export geometry to Open CASCADE.")
SET(BUILD_SHARED ON CACHE BOOL "Build shared libraries (set OFF for wheel packaging)")
SET(BUILD_DOCUMENTATION OFF CACHE BOOL "Build documentation using doxygen")
SET(BUILD_PROFILE OFF CACHE BOOL "Build profiling")

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
