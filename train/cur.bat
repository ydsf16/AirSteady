
set VIDEO_PATH=XXX.webm

ffmpeg -i 'C:\Users\ydsf1\Desktop\F-16 Viper Demo - Harrisburg Airshow 2025.webm' -c:v libx264 -crf 23 -c:a aac -b:a 192k output.mp4