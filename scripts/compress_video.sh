#!/bin/bash

# AirSteady 视频压缩脚本
# 使用 FFmpeg 将视频压缩到 100MB 以内（适配 GitHub 上传限制）

INPUT="resources/AirSteady.mp4"
OUTPUT="resources/AirSteady_compressed.mp4"

# 检查输入文件
if [ ! -f "$INPUT" ]; then
    echo "错误：找不到输入文件 $INPUT"
    exit 1
fi

# 获取视频时长（秒）
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null)
if [ -z "$DURATION" ]; then
    echo "错误：无法获取视频时长"
    exit 1
fi

echo "视频时长：${DURATION}秒"

# 目标文件大小：90MB（留 10MB 余量）
# 计算目标码率 = (文件大小 * 8) / 时长 - 音频码率
# 90MB = 90 * 1024 * 8 kbit
TARGET_SIZE_KBIT=$((90 * 1024 * 8))
AUDIO_BITRATE=128  # kbps
VIDEO_BITRATE=$(( (TARGET_SIZE_KBIT / DURATION) - AUDIO_BITRATE ))

echo "目标视频码率：${VIDEO_BITRATE} kbps"
echo "音频码率：${AUDIO_BITRATE} kbps"

# 压缩视频
echo "开始压缩..."
ffmpeg -i "$INPUT" \
    -c:v libx264 \
    -b:v "${VIDEO_BITRATE}k" \
    -c:a aac \
    -b:a "${AUDIO_BITRATE}k" \
    -movflags +faststart \
    -y \
    "$OUTPUT"

# 检查结果
if [ $? -eq 0 ]; then
    # 显示结果
    ORIGINAL_SIZE=$(stat -c%s "$INPUT" 2>/dev/null || stat -f%z "$INPUT")
    COMPRESSED_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")

    echo ""
    echo "===== 压缩完成 ====="
    echo "原始文件：$(echo "scale=2; $ORIGINAL_SIZE/1024/1024" | bc) MB"
    echo "压缩后：$(echo "scale=2; $COMPRESSED_SIZE/1024/1024" | bc) MB"
    echo "压缩比：$(echo "scale=1; ($ORIGINAL_SIZE-$COMPRESSED_SIZE)*100/$ORIGINAL_SIZE" | bc)%"
    echo ""

    # 检查是否小于 100MB
    if [ $COMPRESSED_SIZE -lt 100000000 ]; then
        echo "✅ 文件大小符合要求（<100MB），可以上传到 GitHub"
    else
        echo "⚠️  文件仍大于 100MB，可以尝试降低码率或使用 CRF 模式"
    fi
else
    echo "❌ 压缩失败"
    exit 1
fi
