@echo off
setlocal enabledelayedexpansion

REM ================== CONFIG ==================
REM Input: 预标注输出目录（包含 images/ 和 labels/）
set "INPUT=E:\AirSteady\code\AirSteady\train\datasets\test"

REM Output: 划分后的数据集目录（会生成 images/train,val 和 labels/train,val）
set "OUTPUT=E:\AirSteady\code\AirSteady\train\datasets\test_split"

REM Validation ratio
set "VAL_RATIO=0.2"

REM Random seed for reproducible split
set "SEED=42"

REM Conda environment name
set "ENV_NAME=air_steady"
REM ============================================

REM cd to current script folder
cd /d "%~dp0"

REM Activate conda env
call conda activate %ENV_NAME%

REM Run split script
python "split_dataset.py" ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --val-ratio %VAL_RATIO% ^
  --seed %SEED%

echo.
echo [DONE] Dataset split finished. Press any key to exit...
pause >nul
