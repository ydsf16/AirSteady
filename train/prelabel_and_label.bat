@echo off
setlocal enabledelayedexpansion

REM =============== CONFIG ===============
REM 1) 视频路径：改成你要打标的视频
set "VIDEO=E:\AirSteady\code\AirSteady\data\f35.mp4"

REM 2) 抽帧间隔：比如 10 表示每隔 10 帧抽一张
set "SKIP=10"

REM 3) YOLO 预标注用的环境 / 模型配置
set "YOLO_ENV=airsteady_gpu"
set "MODEL=yolo11x.pt"
set "IMGSZ=1024"
set "DEVICE=0"

REM 4) LabelImg 环境
set "LABELIMG_ENV=airsteady_gpu"

REM 5) 数据集根目录（预标注 + 标注结果都会放这里）
REM    默认是当前工程下的 \datasets
set "DATASETS_ROOT=%~dp0\datasets"

REM 6) 划分 train/val 的配置
set "VAL_RATIO=0.2"
set "SEED=42"
REM ======================================

REM cd 到脚本所在目录（很重要）
cd /d "%~dp0"

REM 检查视频是否存在
if not exist "%VIDEO%" (
    echo [ERROR] VIDEO not found: %VIDEO%
    echo 请在 bat 开头 CONFIG 区修改 VIDEO 路径。
    pause
    exit /b 1
)

REM 从 VIDEO 中取出文件名（不带扩展），作为数据集名
for %%F in ("%VIDEO%") do set "BASENAME=%%~nF"

REM 预标注输出目录
set "OUTPUT=%DATASETS_ROOT%\%BASENAME%"
set "IMG_DIR=%OUTPUT%\images"
set "LABEL_DIR=%OUTPUT%\labels"
set "CLASSES_FILE=%LABEL_DIR%\classes.txt"

REM 划分后的输出目录（train/val）
set "SPLIT_OUTPUT=%DATASETS_ROOT%\%BASENAME%_split"

echo ========= Prelabel + LabelImg + Split Config =========
echo VIDEO         = %VIDEO%
echo SKIP          = %SKIP%
echo YOLO_ENV      = %YOLO_ENV%
echo MODEL         = %MODEL%
echo IMGSZ         = %IMGSZ%
echo DEVICE        = %DEVICE%
echo LABELIMG_ENV  = %LABELIMG_ENV%
echo DATASETS_ROOT = %DATASETS_ROOT%
echo OUTPUT        = %OUTPUT%
echo IMG_DIR       = %IMG_DIR%
echo LABEL_DIR     = %LABEL_DIR%
echo CLASSES_FILE  = %CLASSES_FILE%
echo SPLIT_OUTPUT  = %SPLIT_OUTPUT%
echo VAL_RATIO     = %VAL_RATIO%
echo SEED          = %SEED%
echo ======================================================
echo.

REM ---------- Step 1: 预标注（抽帧 + YOLO） ----------
echo [STEP 1] Activate YOLO env and run prelabel...
call conda activate %YOLO_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %YOLO_ENV%
    pause
    exit /b 1
)

python "extract_and_prelabel.py" ^
  -v "%VIDEO%" ^
  -o "%OUTPUT%" ^
  -m "%MODEL%" ^
  -s %SKIP% ^
  --imgsz %IMGSZ% ^
  --device %DEVICE%

if errorlevel 1 (
    echo [ERROR] Prelabel script failed.
    pause
    exit /b 1
)

if not exist "%IMG_DIR%" (
    echo [ERROR] No images generated in %IMG_DIR%
    pause
    exit /b 1
)

if not exist "%LABEL_DIR%" (
    echo [ERROR] No labels folder found in %LABEL_DIR%
    pause
    exit /b 1
)

if not exist "%CLASSES_FILE%" (
    echo [WARN] classes.txt not found: %CLASSES_FILE%
    echo        LabelImg YOLO 模式仍可用，但不会显示类别名称。
) else (
    echo [INFO] Found classes.txt: %CLASSES_FILE%
)

echo.
echo [STEP 1 DONE] Prelabel finished. Dataset at: %OUTPUT%
echo.

REM ---------- Step 2: 打开 LabelImg ----------
echo [STEP 2] Activate LabelImg env and start LabelImg...

call conda activate %LABELIMG_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %LABELIMG_ENV%
    pause
    exit /b 1
)

REM LabelImg 参数：
REM   1) image dir
REM   2) classes.txt
REM   3) label dir
labelImg "%IMG_DIR%" "%CLASSES_FILE%" "%LABEL_DIR%"

echo.
echo [STEP 2 DONE] LabelImg closed. Labeled dataset is in:
echo        %OUTPUT%
echo.

REM ---------- Step 3: 划分 train / val ----------
echo [STEP 3] Activate YOLO env and run split_dataset...

call conda activate %YOLO_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to re-activate conda env: %YOLO_ENV%
    pause
    exit /b 1
)

python "split_dataset.py" ^
  --input "%OUTPUT%" ^
  --output "%SPLIT_OUTPUT%" ^
  --val-ratio %VAL_RATIO% ^
  --seed %SEED%

if errorlevel 1 (
    echo [ERROR] split_dataset.py failed.
    pause
    exit /b 1
)

echo.
echo [DONE] All done.
echo        原始标注数据: %OUTPUT%
echo        train/val 划分数据: %SPLIT_OUTPUT%
echo.
echo Press any key to exit...
pause >nul
