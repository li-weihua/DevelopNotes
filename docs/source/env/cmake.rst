CMAKE
=======

cudatoolkit version and supported gpus
----------------------------------------


How to get the version of library found by CMake?
--------------------------------------------------

Ref: https://stackoverflow.com/questions/34138886/how-to-know-version-of-library-found-by-cmake

``<package>_VERSION``


How to build x86 or x64 on Windows from command line with CMAKE?
--------------------------------------------------------------------

Ref: https://stackoverflow.com/questions/28350214/how-to-build-x86-and-or-x64-on-windows-from-command-line-with-cmake

.. code-block:: bash

    cmake -G "Visual Studio 17 2022" -A Win32 -S .. -B "build32"
    cmake -G "Visual Studio 17 2022" -A x64 -S .. -B "build64"
    cmake --build build32 --config Release
    cmake --build build64 --config Release


For simplicity, only build 64bit version:

.. code-block:: bash

    cmake .. -A x64
    cmake --build . --config Release
