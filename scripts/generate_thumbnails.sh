#!/bin/bash
# 从 B 站视频生成封面图（占位符）
# 实际使用时，建议用视频截图或 B 站官方封面

mkdir -p resources

# F22.png - 使用 B 站视频封面
# B 站封面 URL: https://i2.hdslb.com/bfs/archive/${bvid}.jpg
# 但需要提取 bvid，这里用简单占位图

# 生成一个简单占位图（需要 ImageMagick）
if command -v convert &> /dev/null; then
    # F22 封面
    convert -size 640x360 \
        gradient:'#1e3a5f-#3b82f6' \
        -gravity center \
        -pointsize 48 \
        -fill white \
        -annotate 0 "F-22 处理对比\n点击观看" \
        resources/F22.png

    # AirSteady 封面
    convert -size 640x360 \
        gradient:'#1e3a5f-#8b5cf6' \
        -gravity center \
        -pointsize 48 \
        -fill white \
        -annotate 0 "AirSteady 软件演示\n点击观看" \
        resources/AirSteady.png

    echo "封面图已生成到 resources/"
else
    echo "需要安装 ImageMagick: sudo apt-get install imagemagick"
    echo "或者手动创建 resources/F22.png 和 resources/AirSteady.png"
fi
