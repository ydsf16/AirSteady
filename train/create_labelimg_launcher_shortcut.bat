@echo off
setlocal

REM ================ 配置区 =================
REM 桌面快捷方式名称
set "SHORTCUT_NAME=LabelImgLauncher.lnk"

REM 启动的脚本（和本 bat 在同一目录）
set "SCRIPT_NAME=labelimg_launcher.py"

REM 你的 airsteady_gpu 环境里的 pythonw.exe 路径
set "PY_EXE=C:\Users\ydsf1\miniconda3\envs\airsteady_gpu\pythonw.exe"
REM ========================================

REM 当前 bat 所在目录
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

if not exist "%SCRIPT_PATH%" (
    echo [ERROR] 找不到 Python 脚本：%SCRIPT_PATH%
    pause
    exit /b 1
)

if not exist "%PY_EXE%" (
    echo [ERROR] 找不到 pythonw.exe：%PY_EXE%
    echo 请确认 Conda 环境路径是否正确。
    pause
    exit /b 1
)

REM 桌面路径
set "DESKTOP_DIR=%USERPROFILE%\Desktop"
set "SHORTCUT_PATH=%DESKTOP_DIR%\%SHORTCUT_NAME%"

REM VBS 文件路径
set "VBS_FILE=%TEMP%\_mk_labelimg_shortcut.vbs"
del "%VBS_FILE%" 2>nul

REM 一行一行写入 VBS，避免上次那种重定向失效
echo Set WshShell = WScript.CreateObject("WScript.Shell") > "%VBS_FILE%"
echo Set oShellLink = WshShell.CreateShortcut("%SHORTCUT_PATH%") >> "%VBS_FILE%"
echo oShellLink.TargetPath = "%PY_EXE%" >> "%VBS_FILE%"
echo oShellLink.Arguments = " ""%SCRIPT_PATH%"" " >> "%VBS_FILE%"
echo oShellLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%VBS_FILE%"
echo oShellLink.WindowStyle = 1 >> "%VBS_FILE%"
echo oShellLink.IconLocation = "%PY_EXE%,0" >> "%VBS_FILE%"
echo oShellLink.Save >> "%VBS_FILE%"

REM 调用 cscript 执行 VBS
cscript //nologo "%VBS_FILE%"
if errorlevel 1 (
    echo [ERROR] cscript 执行失败，快捷方式可能没有创建成功。
    pause
    exit /b 1
)

del "%VBS_FILE%" 2>nul

echo.
echo [OK] 已在桌面创建快捷方式：
echo     %SHORTCUT_PATH%
echo.
pause
