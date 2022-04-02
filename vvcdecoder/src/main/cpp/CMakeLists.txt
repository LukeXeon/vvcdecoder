# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

set(ANDROID_STL c++_shared)

# Declares and names the project.

project("vvcdeocder_jni")

# Devices using armeabi-v7a are not required to support
# Neon which is why Neon is disabled by default for
# armeabi-v7a build. This flag enables it.
if (${ANDROID_ABI} MATCHES "armeabi-v7a")
    add_compile_options("-mfpu=neon")
    add_compile_options("-marm")
    add_compile_options("-fPIC")
endif ()

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
if (build_type MATCHES "^rel")
    add_compile_options("-O2")
endif ()

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.



# Build vvdec.
add_subdirectory(vvdec)

add_subdirectory(cpufeatures)

include_directories("vvdec/include/vvdec")

add_library( # Sets the name of the library.
        vvcdeocder_jni

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        jni.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        vvcdeocder_jni
        # Links the target library to the log library
        # included in the NDK.
        vvdec
        cpufeatures
        ${log_lib}
        )