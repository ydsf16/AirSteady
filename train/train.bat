@echo off
setlocal enabledelayedexpansion

REM ================== CONFIG ==================
REM Conda environment name
set "ENV_NAME=airsteady"

REM YOLO model weights
set "MODEL=yolo11n.pt"

REM Data yaml (注意这里路径保持你现在的写法)
set "DATA_YAML=E:/AirSteady/code/AirSteady/train/full_dataset.yaml"

REM Training hyperparameters
set "IMG_SIZE=960"
set "EPOCHS=30"
set "BATCH=10"
set "DEVICE=0"

REM Ultralytics train output
set "PROJECT=AirSteady"
set "RUN_NAME=AirSteady80"
REM ============================================

REM cd to current script folder (important for relative paths)
cd /d "%~dp0"

echo ========= YOLO Training Config =========
echo ENV_NAME   = %ENV_NAME%
echo MODEL      = %MODEL%
echo DATA_YAML  = %DATA_YAML%
echo IMG_SIZE   = %IMG_SIZE%
echo EPOCHS     = %EPOCHS%
echo BATCH      = %BATCH%
echo DEVICE     = %DEVICE%
echo PROJECT    = %PROJECT%
echo RUN_NAME   = %RUN_NAME%
echo ========================================
echo.

REM Activate conda env
call conda activate %ENV_NAME%

REM Run training
yolo detect train ^
  model=%MODEL% ^
  data=%DATA_YAML% ^
  imgsz=%IMG_SIZE% ^
  epochs=%EPOCHS% ^
  batch=%BATCH% ^
  device=%DEVICE% ^
  project=%PROJECT% ^
  name=%RUN_NAME% ^
  freeze=10

echo.
echo [DONE] YOLO training finished. Press any key to exit...
pause >nul
