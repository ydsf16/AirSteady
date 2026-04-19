@echo off
REM AirSteady 视频压缩脚本 (Windows 批处理版)
REM 使用方法：
REM 1. 先安装 ffmpeg (推荐：choco install ffmpeg 或从 https://ffmpeg.org/download.html 下载)
REM 2. 双击运行此脚本

cd /d "%~dp0"

echo ========================================
echo AirSteady 视频压缩工具
echo ========================================
echo.

REM 检查 ffmpeg 是否存在
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 ffmpeg
    echo.
    echo 请先安装 ffmpeg:
    echo   方法 1: choco install ffmpeg
    echo   方法 2: 从 https://ffmpeg.org/download.html 下载并添加到 PATH
    echo   方法 3: 使用在线工具压缩 (https://www.freeconvert.com/video-compressor)
    echo.
    pause
    exit /b 1
)

echo [信息] ffmpeg 已找到
echo.

set INPUT=resources\AirSteady.mp4
set OUTPUT=resources\AirSteady_compressed.mp4

if not exist "%INPUT%" (
    echo [错误] 找不到输入文件：%INPUT%
    pause
    exit /b 1
)

echo [信息] 输入文件：%INPUT%
echo [信息] 输出文件：%OUTPUT%
echo.
echo 开始压缩...
echo.

REM 压缩视频
REM -crf 28: 质量参数 (18-28 推荐，数字越大压缩率越高)
REM -c:a aac -b:a 128k: AAC 音频 128kbps
ffmpeg -i "%INPUT%" -c:v libx264 -crf 28 -c:a aac -b:a 128k -movflags +faststart -y "%OUTPUT%"

if %errorlevel% neq 0 (
    echo.
    echo [错误] 压缩失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 压缩完成!
echo ========================================
echo.

REM 显示文件大小
for %%A in ("%INPUT%") do set "ORIGINAL=%%~zA"
for %%A in ("%OUTPUT%") do set "COMPRESSED=%%~zA"

set /a RATIO=(%ORIGINAL%-%COMPRESSED%)*100/%ORIGINAL%

echo 原始大小：%ORIGINAL% 字节
echo 压缩后：%COMPRESSED% 字节
echo 压缩比：%RATIO%%%
echo.

if %COMPRESSED% LSS 100000000 (
    echo [成功] 文件小于 100MB，可以上传到 GitHub
    echo.
    echo 下一步:
    echo   git add resources/
    echo   git commit -m "Add compressed AirSteady demo video"
    echo   git push
) else (
    echo [警告] 文件仍大于 100MB
    echo 尝试使用更激进的压缩参数 (-crf 30 或更高)
)

echo.
pause
