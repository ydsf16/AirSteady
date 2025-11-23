@echo off
setlocal enabledelayedexpansion

REM 切到当前脚本所在文件夹（很重要，确保相对路径正确）
cd /d "%~dp0"

REM ================== 配置区 ==================
set "VIDEO=E:\AirSteady\code\AirSteady\data\f16.mp4"
set "OUTPUT=E:\AirSteady\code\AirSteady\train\datasets\f16"
set "MODEL=yolo11x.pt"
set "SKIP=10"
set "IMGSZ=1024"
set "DEVICE=0"
set "ENV_NAME=air_steady"
REM ==========================================

REM 激活 conda 环境
call conda activate %ENV_NAME%

REM 执行预标注脚本
python "extract_and_prelabel.py" ^
  -v "%VIDEO%" ^
  -o "%OUTPUT%" ^
  -m "%MODEL%" ^
  -s %SKIP% ^
  --imgsz %IMGSZ% ^
  --device %DEVICE%

echo.
echo [DONE] Prelabel finished. Press any key to exit...
pause >nul
