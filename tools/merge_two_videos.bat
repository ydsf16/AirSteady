@echo off
setlocal enabledelayedexpansion

REM =============== CONFIG ===============
REM Dataset paths
set "VIDEO1=C:\Users\ydsf1\AppData\Local\AirSteady\cache\tmp_track.mp4"
set "VIDEO2=C:\Users\ydsf1\AppData\Local\AirSteady\cache\tmp_crop.mp4"

REM 输出目录（不用写文件名，只写文件夹）
set "OUT_DIR=E:\AirSteady\code\AirSteady\data"
REM 基础文件名前缀
set "OUT_PREFIX=merge"
REM ======================================

REM cd to current script folder
cd /d "%~dp0"

REM -------- 生成日期时间 tag：YYYYMMDD_HHMM --------
for /f "tokens=1 delims=." %%a in ('wmic os get LocalDateTime ^| find "."') do (
    set "DT=%%a"
)

REM DT 形如：20251130HHMMSSxxxx
set "DATE_TAG=!DT:~0,8!"
set "TIME_TAG=!DT:~8,4!"  REM HHMM

REM 拼接输出文件名：merge_20251130_1305.mp4
set "OUTPUT=%OUT_DIR%\%OUT_PREFIX%_!DATE_TAG!_!TIME_TAG!.mp4"

echo [INFO] VIDEO1  = %VIDEO1%
echo [INFO] VIDEO2  = %VIDEO2%
echo [INFO] OUTPUT  = %OUTPUT%
echo [INFO] MODE    = auto
echo.

REM 确保输出目录存在
if not exist "%OUT_DIR%" (
    mkdir "%OUT_DIR%"
)

REM Run with parameters (positional args)
python merge_two_videos.py "%VIDEO1%" "%VIDEO2%" "%OUTPUT%" --mode auto

echo.
echo [INFO] closed. Press any key to exit...
pause >nul
