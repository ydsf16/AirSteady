@echo off
setlocal enabledelayedexpansion

REM cd 到脚本所在目录（很重要）
cd /d "%~dp0"

echo [STEP 1] Activate env ...
set "LABELIMG_ENV=airsteady_gpu"
call conda activate %LABELIMG_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %LABELIMG_ENV%
    pause
    exit /b 1
)

echo [STEP 2] RUN ...
python "core\encrypt_assets.py"
python "core\main.py"