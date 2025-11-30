@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM 1) 固定：切到当前脚本所在文件夹（确保相对路径正确）
REM =====================================================
cd /d "%~dp0"

REM =====================================================
REM 2) 基本配置
REM =====================================================
REM Conda 环境名
set "ENV_NAME=airsteady_gpu"

REM YOLO 模型
set "MODEL=yolo11x.pt"

REM 预打标参数
set "SKIP=150"
set "IMGSZ=1024"
set "DEVICE=0"

REM [输入]：需要预打标的视频所在文件夹
set "VIDEO_DIR=E:\AirSteady\code\AirSteady\data\恒庐"

REM [输出]：预打标输出的根目录
set "OUTPUT_BASE_DIR=E:\AirSteady\code\AirSteady\train\datasets\恒庐_prelabel"

REM 支持的视频后缀（不区分大小写）
REM 你可以按需增删
set "VIDEO_EXT_LIST=.mp4 .mov .mkv .avi .wmv .flv .mpg .mpeg .m4v .ts"

REM =====================================================
echo ========= Prelabel Folder Batch Config =========
echo ENV_NAME        = %ENV_NAME%
echo MODEL           = %MODEL%
echo SKIP            = %SKIP%
echo IMGSZ           = %IMGSZ%
echo DEVICE          = %DEVICE%
echo VIDEO_DIR       = %VIDEO_DIR%
echo OUTPUT_BASE_DIR = %OUTPUT_BASE_DIR%
echo VIDEO_EXT_LIST  = %VIDEO_EXT_LIST%
echo ==================================================
echo.

REM 检查输入目录是否存在
if not exist "%VIDEO_DIR%" (
    echo [ERROR] VIDEO_DIR not found: "%VIDEO_DIR%"
    echo 请检查路径后重试。
    pause >nul
    exit /b 1
)

REM =====================================================
REM 3) 激活 conda 环境（只激活一次）
REM =====================================================
call conda activate %ENV_NAME%

REM =====================================================
REM 4) 主循环：遍历文件夹内所有文件，只处理视频文件
REM =====================================================
for %%F in ("%VIDEO_DIR%\*") do (
    REM 如果是目录则跳过
    if exist "%%F\" (
        echo [SKIP] Directory: %%F
    ) else (
        set "IS_VIDEO="
        for %%E in (%VIDEO_EXT_LIST%) do (
            if /I "%%~xF"=="%%E" set "IS_VIDEO=1"
        )

        if not defined IS_VIDEO (
            echo [SKIP] Not video file: %%~nxF
        ) else (
            set "VIDEO_PATH=%%~fF"
            set "VIDEO_NAME=%%~nF"
            set "OUTPUT_DIR=%OUTPUT_BASE_DIR%\!VIDEO_NAME!"

            echo.
            echo -----------------------------------------
            echo [INFO] Start prelabel: %%~nxF
            echo   Video : "!VIDEO_PATH!"
            echo   Output: "!OUTPUT_DIR!"
            echo -----------------------------------------

            python "extract_and_prelabel.py" ^
              -v "!VIDEO_PATH!" ^
              -o "!OUTPUT_DIR!" ^
              -m "%MODEL%" ^
              -s %SKIP% ^
              --imgsz %IMGSZ% ^
              --device %DEVICE%

            if errorlevel 1 (
                echo [WARN] Prelabel FAILED for %%~nxF
            ) else (
                echo [OK] Prelabel SUCCESS for %%~nxF
            )
        )
    )
)

echo.
echo [DONE] All prelabel jobs in folder finished. Press any key to exit...
pause >nul
