IF(BUILD_PYTHON_INTERFACE)

    # ========================================================================
    # Find Python - use modern FindPython for scikit-build-core compatibility
    # ========================================================================
    if(NOT DEFINED SKBUILD)
        find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
        if(NOT DEFINED PYTHON_SITEPACKAGES_DIR)
            execute_process(
                COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
                OUTPUT_VARIABLE PYTHON_SITEPACKAGES_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE TEXGEN_PYTHON_SITEPACKAGES_RESULT)
            if(NOT TEXGEN_PYTHON_SITEPACKAGES_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to query Python site-packages directory")
            endif()
        endif()
    endif()

    # ========================================================================
    # Include directories
    # ========================================================================
    INCLUDE_DIRECTORIES(${Python_INCLUDE_DIRS})
    if(TEXGEN_HAS_P4EST)
        INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../OctreeRefinement/include)
    endif()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE TEXGEN_NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TEXGEN_NUMPY_INCLUDE_RESULT)
    if(NOT TEXGEN_NUMPY_INCLUDE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to query NumPy include directory")
    endif()

    # ========================================================================
    # SWIG Core wrapper
    #
    # Wheel builds use committed generated files by default so users and CI do
    # not need a system SWIG install.  Set TEXGEN_REGENERATE_SWIG=ON only when
    # Core.i or wrapped C++ APIs change. If Core_wrap.cxx is regenerated, keep
    # the _fastdata_extract_snapshot_bundle_direct hook near the method table.
    # ========================================================================
    set(TEXGEN_CORE_SWIG_PY "${CMAKE_CURRENT_SOURCE_DIR}/Core.py")
    set(TEXGEN_CORE_SWIG_CXX "${CMAKE_CURRENT_SOURCE_DIR}/Core_wrap.cxx")

    if(TEXGEN_REGENERATE_SWIG)
        FIND_PACKAGE(SWIG REQUIRED)
        FILE(GLOB CORE_HEADER_FILES ../Core/*.h)

        # Pass -DTEXGEN_USE_P4EST to SWIG when p4est is available.
        set(SWIG_P4EST_FLAG "")
        if(TEXGEN_HAS_P4EST)
            set(SWIG_P4EST_FLAG "-DTEXGEN_USE_P4EST")
        endif()

        ADD_CUSTOM_COMMAND(OUTPUT "${TEXGEN_CORE_SWIG_CXX}" "${TEXGEN_CORE_SWIG_PY}"
            COMMAND "${SWIG_EXECUTABLE}" -c++ -python
                ${SWIG_P4EST_FLAG}
                -outdir "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/Core.i"
            MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/Core.i"
            DEPENDS ${CORE_HEADER_FILES}
            COMMENT "Generating SWIG Core wrapper code...")
        SET_SOURCE_FILES_PROPERTIES("${TEXGEN_CORE_SWIG_CXX}" PROPERTIES GENERATED 1)
    else()
        if(NOT EXISTS "${TEXGEN_CORE_SWIG_CXX}" OR NOT EXISTS "${TEXGEN_CORE_SWIG_PY}")
            MESSAGE(FATAL_ERROR
                "Pre-generated SWIG files are missing. Expected:\n"
                "  ${TEXGEN_CORE_SWIG_CXX}\n"
                "  ${TEXGEN_CORE_SWIG_PY}\n"
                "Install SWIG and configure with -DTEXGEN_REGENERATE_SWIG=ON "
                "to regenerate them.")
        endif()
        MESSAGE(STATUS "Using pre-generated SWIG Core wrapper")
        if(TEXGEN_HAS_P4EST)
            MESSAGE(STATUS "Pre-generated Core wrapper omits p4est; use -DTEXGEN_REGENERATE_SWIG=ON to expose COctreeVoxelMesh")
        endif()
    endif()

    if(COMMAND cmake_policy)
        cmake_policy(SET CMP0003 NEW)
    endif()

    # ========================================================================
    # Build _Core extension module
    # ========================================================================
    ADD_LIBRARY(_Core MODULE "${TEXGEN_CORE_SWIG_CXX}" core_fastdata.cpp)

    # Naming conventions: _Core.pyd (Windows) or _Core.so (Unix)
    SET_TARGET_PROPERTIES(_Core PROPERTIES PREFIX "" DEBUG_POSTFIX "_d")
    SET_TARGET_PROPERTIES(_Core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
    IF(WIN32)
        SET_TARGET_PROPERTIES(_Core PROPERTIES
            SUFFIX ".pyd")
        if(MINGW)
            target_link_options(_Core PRIVATE
                "-Wl,--export-all-symbols"
                "-static"
                "-static-libgcc"
                "-static-libstdc++")
        endif()
    ENDIF()

    IF(APPLE)
        set(_TEXGEN_EXTENSION_RPATH "@loader_path")
    ELSEIF(UNIX)
        set(_TEXGEN_EXTENSION_RPATH "$ORIGIN")
    ENDIF()

    IF(UNIX)
        SET_TARGET_PROPERTIES(_Core PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH 1
            BUILD_WITH_INSTALL_RPATH 1
            INSTALL_RPATH "${_TEXGEN_EXTENSION_RPATH}")
    ENDIF()

    # In SKBUILD (wheel) mode, TexGenCore is STATIC so there must be no
    # dllexport/dllimport.  The PUBLIC definition on TexGenCore should
    # propagate, but we also set it explicitly here to be safe (avoids
    # LNK2019/LNK4217 from __declspec(dllimport) mismatch on MSVC).
    if(DEFINED SKBUILD)
        target_compile_definitions(_Core PRIVATE "CLASS_DECLSPEC=")
    endif()

    # Link to TexGenCore and Python
    TARGET_LINK_LIBRARIES(_Core TexGenCore)
    TARGET_LINK_LIBRARIES(_Core Python::Module)
    target_include_directories(_Core PRIVATE "${TEXGEN_NUMPY_INCLUDE_DIR}")
    target_compile_features(_Core PRIVATE cxx_std_11)

    # ========================================================================
    # Optional fast data extraction provider
    #
    # The direct fast-data extractor is linked into _Core so it sees the same
    # CTextile pointers and TexGenCore singleton as the SWIG wrapper.  The
    # separate _fastdata module is only a CPython facade that delegates to
    # _Core; it deliberately does not link TexGenCore a second time.
    # ========================================================================
    ADD_LIBRARY(_fastdata MODULE fastdata.cpp)
    SET_TARGET_PROPERTIES(_fastdata PROPERTIES PREFIX "" DEBUG_POSTFIX "_d")
    IF(WIN32)
        SET_TARGET_PROPERTIES(_fastdata PROPERTIES SUFFIX ".pyd")
    ENDIF()
    IF(UNIX)
        SET_TARGET_PROPERTIES(_fastdata PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH 1
            BUILD_WITH_INSTALL_RPATH 1
            INSTALL_RPATH "${_TEXGEN_EXTENSION_RPATH}")
    ENDIF()
    TARGET_LINK_LIBRARIES(_fastdata Python::Module)
    target_compile_features(_fastdata PRIVATE cxx_std_11)

    # ========================================================================
    # Python utility modules list
    # ========================================================================
    FILE(GLOB PYTHON_MODULE_FILES Lib/*.py)
    set(PYTEXGEN_INTEROP_MODULE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/gpu_voxelizer.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/material_fields.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/simulation_sample.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/simulation_io.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/acdm_solver.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/torch_periodic_mesher.py")

    # ========================================================================
    # Install targets
    # ========================================================================
    if(DEFINED SKBUILD)
        # =================================================================
        # WHEEL MODE (pip install / python -m build)
        # Install everything into the pytexgen package inside the wheel
        # =================================================================
        set(PYTEXGEN_INSTALL_DIR "pytexgen")

        # The compiled _Core extension module
        install(TARGETS _Core
            LIBRARY DESTINATION ${PYTEXGEN_INSTALL_DIR}
            RUNTIME DESTINATION ${PYTEXGEN_INSTALL_DIR})
        install(TARGETS _fastdata
            LIBRARY DESTINATION ${PYTEXGEN_INSTALL_DIR}
            RUNTIME DESTINATION ${PYTEXGEN_INSTALL_DIR})

        # SWIG-generated Python proxy (Core.py)
        install(FILES "${TEXGEN_CORE_SWIG_PY}"
            DESTINATION ${PYTEXGEN_INSTALL_DIR})

        # Python utility modules (Abaqus.py, Ansys.py, etc.)
        install(FILES ${PYTHON_MODULE_FILES}
            DESTINATION ${PYTEXGEN_INSTALL_DIR})

        # Optional GPU/interoperability modules live in the source
        # compatibility package and are installed together.
        install(FILES ${PYTEXGEN_INTEROP_MODULE_FILES}
            DESTINATION ${PYTEXGEN_INSTALL_DIR})

        # On Windows, also install the TexGenCore DLL if built as shared
        if(WIN32 AND BUILD_SHARED)
            install(TARGETS TexGenCore
                RUNTIME DESTINATION ${PYTEXGEN_INSTALL_DIR})
        endif()

    else()
        # =================================================================
        # LEGACY MODE (cmake --build && cmake --install)
        # Preserves original behaviour for non-pip builds
        # =================================================================
        SET(PYTHON_MODULES "'Abaqus', 'Ansys', 'Completer', 'FlowTex', 'GridFile', 'WiseTex'")
        IF(BUILD_RENDERER)
            SET(PYTHON_MODULES "${PYTHON_MODULES}, 'Renderer'")
        ENDIF()
        IF(BUILD_CASCADE_EXPORT)
            SET(PYTHON_MODULES "${PYTHON_MODULES}, 'Export'")
        ENDIF()

        set(TEXGEN_LEGACY_INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
        FILE(WRITE "${TEXGEN_LEGACY_INIT_PY}" "# __all__ = [${PYTHON_MODULES}]")

        INSTALL(FILES "${TEXGEN_LEGACY_INIT_PY}" "${TEXGEN_CORE_SWIG_PY}"
            DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(FILES ${PYTHON_MODULE_FILES}
            DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(FILES ${PYTEXGEN_INTEROP_MODULE_FILES}
            DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)

        FILE(GLOB PYTHON_SAMPLE_SCRIPTS Scripts/*.py)

        INSTALL(TARGETS _Core
            LIBRARY DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(TARGETS _fastdata
            LIBRARY DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)

        IF(WIN32)
            INSTALL(FILES ${PYTHON_SAMPLE_SCRIPTS}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/Scripts)
            INSTALL(TARGETS _Core
                LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Installer)
            INSTALL(TARGETS _fastdata
                LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Installer)
        ELSE()
            INSTALL(FILES ${PYTHON_SAMPLE_SCRIPTS}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/share/TexGen/Scripts)
        ENDIF()
    endif()

    # ========================================================================
    # Renderer module (only in legacy mode - not included in core wheel)
    # ========================================================================
    IF(BUILD_RENDERER AND NOT DEFINED SKBUILD)
        FIND_PACKAGE(SWIG REQUIRED)
        FILE(GLOB RENDERER_HEADER_FILES ../Renderer/*.h)
        FIND_PACKAGE(VTK 6.3 REQUIRED NO_MODULE)
        INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})

        ADD_CUSTOM_COMMAND(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/Renderer_wrap.cxx"
            COMMAND "${SWIG_EXECUTABLE}" -c++ -python
                -outdir "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/Renderer.i"
            MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/Renderer.i"
            DEPENDS ${RENDERER_HEADER_FILES}
            COMMENT "Generating SWIG Renderer wrapper code...")

        ADD_LIBRARY(_Renderer MODULE Renderer.i Renderer_wrap.cxx)
        SET_SOURCE_FILES_PROPERTIES(Renderer_wrap.cxx PROPERTIES GENERATED 1)
        SET_TARGET_PROPERTIES(_Renderer PROPERTIES PREFIX "" DEBUG_POSTFIX "_d")
        IF(WIN32)
            SET_TARGET_PROPERTIES(_Renderer PROPERTIES SUFFIX ".pyd")
        ENDIF()
        IF(UNIX)
            SET_TARGET_PROPERTIES(_Renderer PROPERTIES COMPILE_FLAGS -Wno-deprecated)
            SET_TARGET_PROPERTIES(_Renderer PROPERTIES
                INSTALL_RPATH_USE_LINK_PATH 1
                BUILD_WITH_INSTALL_RPATH 1
                INSTALL_RPATH "${VTK_LIBRARY_DIRS};${CMAKE_INSTALL_PREFIX}/lib")
            SET_TARGET_PROPERTIES(_Renderer PROPERTIES
                LINK_FLAGS "-Wl,-rpath,${VTK_DIR} -L${VTK_DIR}")
        ENDIF()

        TARGET_LINK_LIBRARIES(_Renderer
            TexGenRenderer
            ${VTK_LIBRARIES}
            Python::Module)

        INSTALL(TARGETS _Renderer
            LIBRARY DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(FILES Renderer.py
            DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        IF(WIN32)
            INSTALL(TARGETS _Renderer
                LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Installer)
        ENDIF()
    ENDIF()

    # ========================================================================
    # Export module (only in legacy mode - not included in core wheel)
    # ========================================================================
    IF(BUILD_CASCADE_EXPORT AND NOT DEFINED SKBUILD)
        FIND_PACKAGE(SWIG REQUIRED)
        SET(EXPORT_HEADER_FILES ../Export/Exporter.h)

        ADD_CUSTOM_COMMAND(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/Export_wrap.cxx"
            COMMAND "${SWIG_EXECUTABLE}" -c++ -python
                -outdir "${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/Export.i"
            MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/Export.i"
            DEPENDS ${EXPORT_HEADER_FILES}
            COMMENT "Generating SWIG Export wrapper code...")

        ADD_LIBRARY(_Export MODULE Export.i Export_wrap.cxx)
        SET_SOURCE_FILES_PROPERTIES(Export_wrap.cxx PROPERTIES GENERATED 1)
        SET_TARGET_PROPERTIES(_Export PROPERTIES PREFIX "" DEBUG_POSTFIX "_d")
        IF(WIN32)
            SET_TARGET_PROPERTIES(_Export PROPERTIES SUFFIX ".pyd")
        ENDIF()
        IF(UNIX)
            SET_TARGET_PROPERTIES(_Export PROPERTIES COMPILE_FLAGS -Wno-deprecated)
            SET_TARGET_PROPERTIES(_Export PROPERTIES
                INSTALL_RPATH_USE_LINK_PATH 1
                BUILD_WITH_INSTALL_RPATH 1
                INSTALL_RPATH "${OPENCASCADE_LIBRARY_DIR};${CMAKE_INSTALL_PREFIX}/lib")
        ENDIF()

        TARGET_LINK_LIBRARIES(_Export
            TexGenExport
            Python::Module)

        INSTALL(TARGETS _Export
            LIBRARY DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        INSTALL(FILES Export.py
            DESTINATION ${PYTHON_SITEPACKAGES_DIR}/TexGen)
        IF(WIN32)
            INSTALL(TARGETS _Export
                LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../Installer)
        ENDIF()
    ENDIF()

ENDIF(BUILD_PYTHON_INTERFACE)
