@echo off
setlocal

rem ====== config vcpkg path ======
set "VCPKG_ROOT=C:\Users\Admin\vcpkg"

if not exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo ERROR: vcpkg.exe not found at "%VCPKG_ROOT%".
    echo Please check VCPKG_ROOT path.
    pause
    exit /b 1
)

rem ====== paths ======
set "SRC_DIR=%~dp0"
set "BUILD_DIR=%SRC_DIR%build"

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

echo [INFO] Source dir : "%SRC_DIR%"
echo [INFO] Build  dir : "%BUILD_DIR%"
echo [INFO] VCPKG_ROOT : "%VCPKG_ROOT%"
echo.

rem ====== configure ======
echo [INFO] Configuring CMake project...
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

if errorlevel 1 (
    echo.
    echo [ERROR] CMake configure failed.
    pause
    exit /b 1
)

rem ====== build ======
echo.
echo [INFO] Building Release configuration...
cmake --build "%BUILD_DIR%" --config Release -- /m

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

rem ====== done ======
echo.
echo [INFO] Build succeeded.

set "EXE_PATH=%BUILD_DIR%\Release\AirSteady.exe"
if exist "%EXE_PATH%" (
    echo [INFO] Executable: "%EXE_PATH%"
) else (
    echo [WARN] AirSteady.exe not found under "%BUILD_DIR%\Release".
)

echo.
pause
endlocal
