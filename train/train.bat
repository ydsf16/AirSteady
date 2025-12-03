@echo off
setlocal enabledelayedexpansion

REM ================== CONFIG ==================
REM Conda environment name
set "ENV_NAME=airsteady_gpu"

REM YOLO model weights
set "MODEL=yolo11n.pt"

REM Data yaml
@REM set "DATA_YAML=E:/AirSteady/code/AirSteady/train/full_dataset.yaml"
set "DATA_YAML=E:/AirSteady/code/AirSteady/train/sample_dataset.yaml"

REM Training hyperparameters
set "IMG_SIZE=960"
set "EPOCHS=40"
set "BATCH=15"
set "DEVICE=0"
set "WORKERS=2"

REM Ultralytics train output
set "PROJECT=AirSteady"
set "RUN_NAME=AirSteady_960_bs15_e40"
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
echo WORKERS    = %WORKERS%
echo PROJECT    = %PROJECT%
echo RUN_NAME   = %RUN_NAME%
echo PIN_MEMORY = false
echo ========================================
echo.

REM Activate conda env
call conda activate %ENV_NAME%

REM ===== 关键：这里是 cmd 写法，真正关掉 PIN_MEMORY =====
set PIN_MEMORY=false

REM Run training
yolo train ^
  model=%MODEL% ^
  data=%DATA_YAML% ^
  imgsz=%IMG_SIZE% ^
  epochs=%EPOCHS% ^
  batch=%BATCH% ^
  device=%DEVICE% ^
  project=%PROJECT% ^
  name=%RUN_NAME% ^
  workers=%WORKERS% ^
  patience=15 ^
  freeze=10

echo.
echo [DONE] YOLO training finished. Press any key to exit...
pause >nul
