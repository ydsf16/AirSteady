@echo off
REM ============================================
REM 自动创建虚拟环境 + 安装依赖 + PyInstaller 打包
REM main.py 在 core\ 目录下，model\ 里面有 model.pt
REM ============================================

REM 1) 切到当前 bat 所在目录（项目根目录）
cd /d "%~dp0"

@REM echo [1/4] 检查 / 创建虚拟环境 .venv ...
@REM if not exist .venv (
@REM     python -m venv .venv
@REM )

@REM echo [2/4] 激活虛拟环境 ...
@REM call .venv\Scripts\activate.bat

@REM echo [3/4] 安装依赖 (ultralytics, PySide6, pyinstaller) ...
@REM REM 如果你有 requirements.txt，也可以改成: pip install -r requirements.txt
@REM pip install --upgrade pip
@REM pip install ultralytics PySide6 pyinstaller

echo [4/4] 使用 PyInstaller 打包 ...

REM 说明：
REM --name AirSteady          生成的 exe 名字
REM --onefile                 打成一个单独的 exe
REM --noconsole               不显示黑色控制台窗口（如果要看log，可以去掉这个）
REM --collect-all PySide6     自动收集 Qt 相关插件
REM --add-data "core\model;core/model"
REM     把 core\model 整个文件夹打包进去，
REM     运行时访问路径是 core/model/...

pyinstaller ^
  --name AirSteady ^
  --noconsole ^
  --collect-all PySide6 ^
  --add-data "core\model;model" ^
  core\main.py

echo.
echo 打包完成，exe 在: dist\AirSteady.exe
echo 按任意键退出...
pause >nul
