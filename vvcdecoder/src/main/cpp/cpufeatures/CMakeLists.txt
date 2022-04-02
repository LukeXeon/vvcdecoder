# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

set(ANDROID_STL c++_shared)

# Declares and names the project.

project("cpufeatures")

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)
if (build_type MATCHES "^rel")
    add_compile_options("-O2")
endif ()

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

# Build vvdec.

add_library( # Sets the name of the library.
        cpufeatures

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        cpu-features.c
        )