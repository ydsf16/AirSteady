@echo off
setlocal

@echo off
chcp 65001 >nul
setlocal

REM ============================================
REM AirSteady - Trial Build (with time limit)
REM ============================================

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

set SRC_DIR=core
set BUILD_SRC=build_trial_src
set OUTDIR=dist

echo.
echo ========= [Trial] Step 1/5 清理构建目录 =========
if exist "%BUILD_SRC%" rd /s /q "%BUILD_SRC%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
mkdir "%BUILD_SRC%" >nul 2>&1

echo.
echo ========= [Trial] Step 2/5 复制源码到 %BUILD_SRC% =========
xcopy "%SRC_DIR%" "%BUILD_SRC%\core" /E /I /Y >nul
if errorlevel 1 (
    echo [ERROR] 复制源码失败，终止构建。
    goto :END
)

echo.
echo ========= [Trial] Step 3/5 加密模型文件 (encrypt_assets.py) =========
python "%SRC_DIR%\encrypt_assets.py"
if errorlevel 1 (
    echo [ERROR] encrypt_assets.py 执行失败，终止构建。
    goto :END
)

echo.
echo ========= [Trial] Step 4/5 使用 PyArmor 为核心模块加壳 =========
REM 这里只保护核心逻辑，不动 UI 代码，避免莫名问题
pyarmor gen ^
  -O "%BUILD_SRC%\core" ^
  "%BUILD_SRC%\core\algorithm.py" ^
  "%BUILD_SRC%\core\license_guard.py" ^
  "%BUILD_SRC%\core\asset_loader.py"
if errorlevel 1 (
    echo [ERROR] PyArmor gen 失败，终止构建。
    goto :END
)

echo.
echo ========= [Trial] Step 5/5 使用 PyInstaller 打 Trial EXE =========
pyinstaller ^
  --noconfirm ^
  --clean ^
  --name AirSteady_Trial ^
  --windowed ^
  --paths "%BUILD_SRC%\core" ^
  --icon "AirSteadyICO.ico" ^
  ^
  --add-data "core\assets\assets.enc;assets" ^
  --add-data "core\assets\douyin_qr.jpg;assets" ^
  --add-data "core\assets\wechat_qr.jpg;assets" ^
  --add-data "core\bin\ffmpeg.exe;bin" ^
  ^
  --hidden-import=algorithm ^
  --hidden-import=license_guard ^
  --hidden-import=asset_loader ^
  --hidden-import=pyarmor_runtime ^
  ^
  --hidden-import=ultralytics ^
  ^
  --hidden-import=cryptography ^
  --hidden-import=cryptography.hazmat ^
  --hidden-import=cryptography.hazmat.primitives ^
  --hidden-import=cryptography.hazmat.primitives.ciphers ^
  --hidden-import=cryptography.hazmat.primitives.ciphers.aead ^
  ^
  --collect-submodules=ultralytics ^
  --collect-submodules=pyarmor_runtime ^
  --collect-submodules=cryptography ^
  ^
  "%BUILD_SRC%\core\main.py"


if errorlevel 1 (
    echo [ERROR] PyInstaller 构建失败。
    goto :END
)

echo.
echo ========= ✅ Trial Build Completed =========
echo 生成文件: dist\AirSteady_Trial.exe

:END
echo.
pause
endlocal
