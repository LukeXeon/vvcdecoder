# Top level CMakeLists.txt
#
# minimum required cmake version
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)

# Devices using armeabi-v7a are not required to support
# Neon which is why Neon is disabled by default for
# armeabi-v7a build. This flag enables it.
if (${ANDROID_ABI} MATCHES "armeabi-v7a")
    add_compile_options("-mfpu=neon")
    add_compile_options("-marm")
    add_compile_options("-fPIC")
endif ()

set(ANDROID_STL c++_shared)

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
if (build_type MATCHES "^rel")
    add_compile_options("-O2")
endif ()

# set cmake policy
if (NOT CMAKE_VERSION VERSION_LESS 3.12.0)
    # Use latest policy
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif ()

# project name
project(vvdec VERSION 1.4.0)
# set alternative version numbering for release candidates
#set( PROJECT_VERSION_RC rc1 )
if (PROJECT_VERSION_RC)
    set(PROJECT_VERSION "${PROJECT_VERSION}-rc${PROJECT_VERSION_RC}")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
message(STATUS "CMAKE_MODULE_PATH: updating module path to: ${CMAKE_MODULE_PATH}")

# we enable x86 intrinsics for all target architectures, because they are implemented through simd-everywhere on non-x86
set(VVDEC_ENABLE_X86_SIMD TRUE CACHE BOOL "enable x86 intrinsics")

include(vvdecCompilerSupport)

# enable sse4.1 build for all source files for gcc and clang
if (VVDEC_ENABLE_X86_SIMD)
    if (UNIX OR MINGW)
        # when building for non-x86, but emulating simd using simd-everywhere (e.g. on ARM),
        # the x86-compiler flags are not understood by the compiler
        set_if_compiler_supports_flag(FLAG_msse41 -msse4.1)
        add_compile_options(${FLAG_msse41})
    endif ()

    if (NOT MSVC AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten")
        check_missing_intrinsics()
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_SIMD_X86")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTARGET_SIMD_X86")
endif ()

# This potentially brings performance improvements for simd-everywhere, but I haven't
# seen any, so it's disabled for now (GH)
#set_if_compiler_supports_flag( FLAG_openmp_simd -fopenmp-simd )
#add_compile_options( ${FLAG_openmp_simd} )

if (${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten")
    set(VVDEC_TARGET_WASM TRUE)

    add_compile_options(-pthread)
    add_link_options(
            --bind
            -sWASM_BIGINT
            -sINITIAL_MEMORY=1200MB # this should be enough for FullHD decoding
            -sINVOKE_RUN=0          # don't call main() automatically
            -sUSE_PTHREADS
            #-sPROXY_TO_PTHREAD
            #-sPTHREAD_POOL_SIZE=20
            -sMINIFY_HTML=0
            -sMODULARIZE
            -sEXPORT_NAME=CreateVVdeC
            -sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,getValue,setValue,_malloc,callMain,FS]
            -sEXPORTED_FUNCTIONS=@${CMAKE_CURRENT_SOURCE_DIR}/source/Lib/vvdec/wasm_exported_functions.json
            #    --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/ext/bitstreams/AMVR_B_HHI_3/AMVR_B_HHI_3.bit@in.bit
    )

    if (VVDEC_ENABLE_X86_SIMD)
        add_compile_options(-msimd128)            # needed for all SIMD code, but breaks running in nodejs (firefox & chrome do work)

        add_compile_definitions(TARGET_SIMD_WASM) # this is currently only used by a single helper function vvdec_get_RGBA_frame()
        # optimized specifically in WASM-SIMD. This has no effect on normal codec operations.
    endif ()
endif ()


if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # set exception handling
    if (MSVC)
        add_compile_options("/EHsc")
    endif ()

    # vvdec embedded by superproject, always include source/Lib/vvdec  as first component
    list(PREPEND ${PROJECT_NAME}_ADD_SUBDIRECTORIES "source/Lib/vvdec")
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_ADD_SUBDIRECTORIES)
    message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}: ${PROJECT_NAME} embedded, subdirectories to be added: ${${PROJECT_NAME}_ADD_SUBDIRECTORIES}")
    # add subdirectories the superproject asked for
    foreach (subdir IN LISTS ${PROJECT_NAME}_ADD_SUBDIRECTORIES)
        add_subdirectory(${subdir})
    endforeach ()

    return()
endif ()

# enable or disable bitstream download
set(VVDEC_ENABLE_BITSTREAM_DOWNLOAD OFF CACHE BOOL "Enable or disable bitstream download")

# enable local bitstream download
set(VVDEC_ENABLE_LOCAL_BITSTREAM_DOWNLOAD OFF CACHE BOOL "Enable or disable local bitstream download")

# enable install target
set(VVDEC_ENABLE_INSTALL ON CACHE BOOL "Enable or disable install target")


set(VVDEC_ENABLE_LINK_TIME_OPT ON CACHE BOOL "Enable link time optimization for release and profile builds")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    # enable address sanitizer
    set(VVDEC_USE_ADDRESS_SANITIZER OFF CACHE BOOL "Enable or disable address sanitizer")

    # add -march=native
    set(VVDEC_OPT_TARGET_ARCH "" CACHE STRING "Enable or disable building with architecture specific optimization")
endif ()

# set VVDEC_OUTPUT_DIR_POSTFIX
if (BUILD_SHARED_LIBS)
    set(VVDEC_OUTPUT_DIR_POSTFIX shared)
else ()
    set(VVDEC_OUTPUT_DIR_POSTFIX static)
endif ()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}")

# enable or disable Intel Vtune ITT Tracing
#if( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
#    set( VVDEC_ENABLE_ITT ON CACHE BOOL "Enable Intel Runtime Support for Profiling" )
#else()
set(VVDEC_ENABLE_ITT OFF CACHE BOOL "Enable Intel Runtime Support for Profiling")
#endif()

# set default CMAKE_BUILD_TYPE to Release if not set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

#set( SET_ENABLE_TRACING OFF CACHE BOOL "Set ENABLE_TRACING as a compiler flag" )
#set( ENABLE_TRACING OFF CACHE BOOL "If SET_ENABLE_TRACING is on, it will be set to this value" )

# set c++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile everything position independent (even static libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# set verbose compile options
#set( CMAKE_VERBOSE_MAKEFILE ON )

# use folders in IDEs for projects (e.g. lib sample app test)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Enable multithreading
find_package(Threads REQUIRED)

# set _WIN32_WINNT
if (WIN32)
    # set _WIN32_WINT version global
    add_definitions(-D_WIN32_WINNT=0x0600)
endif ()

# enable parallel build for Visual Studio
if (MSVC)
    # add compile options
    add_compile_options("/MP")
    add_compile_options("/EHsc")
endif ()

# set address sanitizer compiler arguments
if (VVDEC_USE_ADDRESS_SANITIZER)
    # add compile options
    add_compile_options("-fsanitize=address")
    add_link_options("-fsanitize=address")
endif ()

if (VVDEC_ENABLE_X86_SIMD)
    if ((UNIX OR MINGW) AND VVDEC_OPT_TARGET_ARCH)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${VVDEC_OPT_TARGET_ARCH} -mtune=${VVDEC_OPT_TARGET_ARCH}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${VVDEC_OPT_TARGET_ARCH} -mtune=${VVDEC_OPT_TARGET_ARCH}")
    endif ()
endif ()

if (VVDEC_ENABLE_LINK_TIME_OPT)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
endif ()

# use ccache
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    message(STATUS "ccache found. using it.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif ()

if (VVDEC_ENABLE_ITT)
    # search for Intel VTune
    if (MSVC)
        set(VVDEC_ITT_PATH "c:/Program Files (x86)/IntelSWTools/VTune Amplifier/" CACHE STRING "Path to the installation directory of Intel VTunes")
    elseif (APPLE)
        message(WARNING "Not yet supported on Mac OS X")
    elseif (UNIX OR MINGW)
        if (EXISTS "/opt/intel/vtune_profiler")
            set(VVDEC_ITT_PATH "/opt/intel/vtune_profiler" CACHE STRING "Path to the installation directory of Intel VTunes")
        else ()
            set(VVDEC_ITT_PATH "/opt/intel/vtune_amplifier" CACHE STRING "Path to the installation directory of Intel VTunes")
        endif ()
    endif ()

    if (EXISTS ${VVDEC_ITT_PATH}/lib64 AND EXISTS ${VVDEC_ITT_PATH}/include)
        set(INTEL_ITT_LINK_TARGET INTEL_ITT)
        add_library(${INTEL_ITT_LINK_TARGET} STATIC IMPORTED GLOBAL)
        if (MSVC)
            set_target_properties(${INTEL_ITT_LINK_TARGET} PROPERTIES IMPORTED_LOCATION ${VVDEC_ITT_PATH}/lib64/libittnotify.lib)
        elseif (APPLE)
            # not supported
        elseif (UNIX OR MINGW)
            set_target_properties(${INTEL_ITT_LINK_TARGET} PROPERTIES IMPORTED_LOCATION ${VVDEC_ITT_PATH}/lib64/libittnotify.a)
            set_target_properties(${INTEL_ITT_LINK_TARGET} PROPERTIES INTERFACE_LINK_LIBRARIES dl)
        endif ()

        # set include directory. relative paths do not work.
        set_target_properties(${INTEL_ITT_LINK_TARGET} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${VVDEC_ITT_PATH}/include)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTRACE_ENABLE_ITT")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRACE_ENABLE_ITT")
    else ()
        message(WARNING "VVDEC_ITT_PATH ${VVDEC_ITT_PATH} not found, ignoring option VVDEC_ENABLE_ITT")
    endif ()
endif ()

add_subdirectory("source/Lib/vvdec")

if (UNIX AND NOT VVDEC_USE_ADDRESS_SANITIZER AND
(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    # run Debug and RelWithDebInfo test suite in GDB and collect backtrace on crashes
    set(VVDEC_TESTS_DEBUGGER_COMMAND gdb --batch --return-child-result --nx
            -iex "set print thread-events off"
            -iex "set index-cache on"
            -ex "run"
            -ex "info locals"
            -ex "backtrace"
            -ex "quit"
            --args)
else ()
    set(VVDEC_TESTS_DEBUGGER_COMMAND "")
endif ()