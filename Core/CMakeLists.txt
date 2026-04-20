FILE(GLOB CORE_SOURCE_FILES *.cpp)
FILE(GLOB CORE_HEADER_FILES *.h)

IF(COMMAND CMAKE_POLICY)
    CMAKE_POLICY(SET CMP0003 NEW)
ENDIF()

# ============================================================================
# Build TexGenCore library
# In SKBUILD (wheel) mode, always build STATIC so everything links into _Core
# ============================================================================
if(DEFINED SKBUILD)
    ADD_LIBRARY(TexGenCore STATIC ${CORE_SOURCE_FILES} ${CORE_HEADER_FILES}
        ../units/units.c ../units/units.h)
    # PIC is needed even for static lib because it links into a shared _Core.so
    SET_TARGET_PROPERTIES(TexGenCore PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    IF(BUILD_SHARED)
        ADD_LIBRARY(TexGenCore SHARED ${CORE_SOURCE_FILES} ${CORE_HEADER_FILES}
            ../units/units.c ../units/units.h)
    ELSE()
        ADD_LIBRARY(TexGenCore STATIC ${CORE_SOURCE_FILES} ${CORE_HEADER_FILES}
            ../units/units.c ../units/units.h)
    ENDIF()
endif()

# ============================================================================
# Precompiled headers (MSVC only, not in SKBUILD mode where Ninja is used)
# The /Yu /Yc flags require a specific build order that Ninja doesn't
# guarantee by default. Disable PCH in wheel builds for reliability.
# ============================================================================
IF(WIN32 AND MSVC AND NOT DEFINED SKBUILD)
    SET_SOURCE_FILES_PROPERTIES(${CORE_SOURCE_FILES}
        PROPERTIES COMPILE_FLAGS /Yu"PrecompiledHeaders.h")
    SET_SOURCE_FILES_PROPERTIES(PrecompiledHeaders.cpp
        PROPERTIES COMPILE_FLAGS /Yc"PrecompiledHeaders.h")
ENDIF()

# ============================================================================
# Symbol visibility: CLASS_DECLSPEC handling
#
# In legacy shared-library builds, -DEXPORT makes CLASS_DECLSPEC resolve to
# __declspec(dllexport) inside TexGenCore, while consumers (without EXPORT
# defined) see __declspec(dllimport).  This is correct for DLLs.
#
# In SKBUILD (wheel) mode TexGenCore is STATIC, so there must be NO
# dllexport/dllimport at all.  We pre-define CLASS_DECLSPEC as empty (before
# the #ifndef in Misc.h) and propagate it to _Core via PUBLIC.
# ============================================================================
if(DEFINED SKBUILD)
    target_compile_definitions(TexGenCore PUBLIC "CLASS_DECLSPEC=")
else()
    ADD_DEFINITIONS(-DEXPORT)
endif()

# ============================================================================
# Link dependencies
# ============================================================================
TARGET_LINK_LIBRARIES(TexGenCore triangle triangle-api Octree TinyXML CSparse tetgenlib)

# Link OpenMP if available (parallelizes CTextile::GetPointInformation hot loop)
IF(OpenMP_CXX_FOUND)
    TARGET_LINK_LIBRARIES(TexGenCore OpenMP::OpenMP_CXX)
ENDIF()

TARGET_INCLUDE_DIRECTORIES(TexGenCore PUBLIC ../OctreeRefinement/include)

# ============================================================================
# p4est/sc support - enables COctreeVoxelMesh
#
# In SKBUILD (wheel) mode: p4est is NOT available, so we compile without it.
# In legacy mode: we look for prebuilt p4est/sc libraries in OctreeRefinement/.
# The TEXGEN_USE_P4EST define guards all p4est-dependent code.
# ============================================================================
set(TEXGEN_HAS_P4EST OFF)
set(TEXGEN_HAS_P4EST OFF PARENT_SCOPE)

if(NOT DEFINED SKBUILD)
    IF(WIN32)
        set(_P4EST_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../OctreeRefinement/libp4est.lib")
        set(_SC_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../OctreeRefinement/libsc.lib")
        if(EXISTS "${_SC_LIB}" AND EXISTS "${_P4EST_LIB}")
            TARGET_LINK_LIBRARIES(TexGenCore ${_SC_LIB} ${_P4EST_LIB})
            set(TEXGEN_HAS_P4EST ON PARENT_SCOPE)
            set(TEXGEN_HAS_P4EST ON)
            message(STATUS "Found p4est: ${_P4EST_LIB}")
            message(STATUS "Found sc: ${_SC_LIB}")
        else()
            message(WARNING "p4est/sc .lib not found in OctreeRefinement/ - "
                            "OctreeVoxelMesh will be disabled")
        endif()
    ELSEIF(UNIX)
        find_library(P4EST_LIB NAMES p4est p4est-2.0
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../OctreeRefinement
            NO_DEFAULT_PATH)
        find_library(SC_LIB NAMES sc sc-2.0
            PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../OctreeRefinement
            NO_DEFAULT_PATH)
        if(P4EST_LIB AND SC_LIB)
            TARGET_LINK_LIBRARIES(TexGenCore ${P4EST_LIB} ${SC_LIB})
            set(TEXGEN_HAS_P4EST ON PARENT_SCOPE)
            set(TEXGEN_HAS_P4EST ON)
            message(STATUS "Found p4est: ${P4EST_LIB}")
            message(STATUS "Found sc: ${SC_LIB}")
        else()
            message(WARNING "p4est/sc libraries not found in OctreeRefinement/ - "
                            "OctreeVoxelMesh will be disabled")
        endif()
    ENDIF()
else()
    message(STATUS "SKBUILD mode: skipping prebuilt p4est/sc (OctreeVoxelMesh disabled)")
endif()

if(TEXGEN_HAS_P4EST)
    TARGET_COMPILE_DEFINITIONS(TexGenCore PUBLIC TEXGEN_USE_P4EST)
endif()

# ============================================================================
# Install targets
# ============================================================================
if(NOT DEFINED SKBUILD)
    # Legacy install mode
    INCLUDE(../Python/FindPythonSitePackages.cmake)
    IF(WIN32)
        INSTALL(TARGETS TexGenCore RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
        INSTALL(TARGETS TexGenCore RUNTIME DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(TARGETS TexGenCore RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Installer)
    ELSE()
        IF(BUILD_SHARED)
            INSTALL(TARGETS TexGenCore LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        ELSE()
            INSTALL(TARGETS TexGenCore ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        ENDIF()
    ENDIF()
endif()
# In SKBUILD mode, _Core links TexGenCore statically - no separate install needed
