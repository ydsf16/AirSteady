@echo off
setlocal enabledelayedexpansion

REM cd 到当前 bat 所在目录
cd /d "%~dp0"

REM 激活 conda 环境（要求里面已经安装：PySide6、ultralytics、labelImg）
set "ENV_NAME=airsteady"
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: %ENV_NAME%
    pause
    exit /b 1
)

REM 启动 GUI
python "label_pipeline_gui.py"

echo.
echo [INFO] GUI 已退出。按任意键关闭窗口...
pause >nul
