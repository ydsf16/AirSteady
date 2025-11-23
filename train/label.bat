@echo off
setlocal enabledelayedexpansion

REM =============== CONFIG ===============
REM Conda environment for LabelImg
set "ENV_NAME=labelimg_env"

REM Dataset paths
set "IMG_DIR=E:\AirSteady\code\AirSteady\train\datasets\f16\images"
set "LABEL_DIR=E:\AirSteady\code\AirSteady\train\datasets\f16\labels"
set "CLASSES_FILE=E:\AirSteady\code\AirSteady\train\datasets\classes.txt"
REM ======================================

REM cd to current script folder
cd /d "%~dp0"

echo ========= LabelImg Config =========
echo ENV_NAME    = %ENV_NAME%
echo IMG_DIR     = %IMG_DIR%
echo CLASSES_FILE= %CLASSES_FILE%
echo LABEL_DIR   = %LABEL_DIR%
echo ===================================
echo.

REM Activate conda env
call conda activate %ENV_NAME%

REM Run LabelImg with parameters:
REM   1) image dir
REM   2) predefined classes.txt
REM   3) save label dir
labelImg "%IMG_DIR%" "%CLASSES_FILE%" "%LABEL_DIR%"

echo.
echo [INFO] LabelImg closed. Press any key to exit...
pause >nul
