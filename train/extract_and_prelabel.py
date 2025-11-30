#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抽帧 + YOLOv11 预标注 一体脚本（命令行版）

功能：
1. 从视频中按固定间隔抽帧到 output_root/images
2. 用 YOLO 模型对每张图做检测
3. 把预测结果保存为 YOLO txt 到 output_root/labels
   （class_id x_center y_center width height，全部是归一化坐标）
4. 同时在 output_root/labels 下生成 classes.txt，便于 LabelImg YOLO 模式使用

依赖：
    pip install ultralytics opencv-python

用法示例（在 airsteady_yolo 环境中）：

    python extract_and_prelabel.py ^
        -v "D:/AirSteady/data/your_video.mp4" ^
        -o "D:/AirSteady/labelimg_dataset" ^
        -m "yolo11s.pt" ^
        -s 10 ^
        --imgsz 960 ^
        --device 0
"""

import argparse
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


def extract_frames(video_path: Path, images_dir: Path, skip: int) -> List[Path]:
    """
    从视频中按间隔抽帧，返回所有保存的帧路径列表。

    参数：
        video_path: 输入视频路径
        images_dir: 抽帧图片输出目录
        skip: 抽帧间隔，例如 skip=10 表示每隔 10 帧保存一张
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 无法打开视频文件: {video_path}")

    frame_idx = 0
    saved_paths: List[Path] = []

    print(f"[INFO] 开始抽帧：每隔 {skip} 帧保存一张 ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            out_path = images_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)

        frame_idx += 1

    cap.release()
    print(f"[INFO] 抽帧完成，共保存 {len(saved_paths)} 张图片到 {images_dir}")
    return saved_paths


def yolo_prelabel(
    image_paths: List[Path],
    labels_dir: Path,
    model_path: Path,
    imgsz: int,
    device: str,
):
    """
    使用 YOLO 模型对所有图片做预标注，输出 YOLO txt，并在 labels_dir 下生成 classes.txt。

    参数：
        image_paths: 所有图片路径列表
        labels_dir: 标签输出目录
        model_path: YOLO 模型路径（yolo11s.pt / 自己微调的 .pt）
        imgsz: 推理分辨率
        device: 设备，如 "0"（第 0 块 GPU）或 "cpu"
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 加载 YOLO 模型: {model_path}")
    model = YOLO(str(model_path))

    # === 先根据模型的 names 写出 classes.txt，给 LabelImg 用 ===
    try:
        names = model.names  # dict: {0: 'person', 1: 'bicycle', ...}
        if isinstance(names, dict) and len(names) > 0:
            classes_path = labels_dir / "classes.txt"
            # 按类别 id 排序，一行一个类名
            lines = [names[i] for i in sorted(names.keys())]
            classes_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[INFO] 已生成 classes.txt: {classes_path}")
        else:
            print("[WARN] 模型未提供 names 字典，无法自动生成 classes.txt")
    except Exception as e:
        print(f"[WARN] 生成 classes.txt 失败: {e}")

    total = len(image_paths)
    print(f"[INFO] 开始预标注，共 {total} 张图片 ...")

    for i, img_path in enumerate(image_paths):
        results = model(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        label_path = labels_dir / f"{img_path.stem}.txt"
        lines = []

        if boxes is not None and len(boxes) > 0:
            # xywhn: 归一化后的 (x_center, y_center, w, h)
            xywhn = boxes.xywhn.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for cls_id, (xc, yc, w, h) in zip(cls, xywhn):
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # 没检测到目标也写一个（可以是空文件）
        label_path.write_text("\n".join(lines), encoding="utf-8")

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"[INFO] 预标注进度: {i + 1}/{total}")

    print(f"[INFO] 预标注完成，标签已保存到 {labels_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从视频抽帧并用 YOLOv11 做预标注，"
            "生成可直接导入 LabelImg 的 images/ + labels/ 目录（含 classes.txt）。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="输入视频路径，例如 D:/AirSteady/data/your_video.mp4",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=str,
        help="输出根目录，将在其中创建 images/ 和 labels/ 子目录",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo11s.pt",
        help="YOLO 模型路径（默认使用官方 yolo11s.pt，COCO 80 类）",
    )
    parser.add_argument(
        "-s",
        "--skip",
        type=int,
        default=10,
        help="抽帧间隔：skip=10 表示每隔 10 帧保存一张图片",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO 推理分辨率 imgsz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='推理设备：GPU 写 "0"、"1"...；CPU 写 "cpu"',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 参数完整性检查
    if not args.video or not args.output_root:
        print("\n[ERROR] 必须指定视频路径和输出目录。示例用法：\n")
        print(
            "  python extract_and_prelabel.py "
            '-v "D:/AirSteady/data/your_video.mp4" '
            '-o "D:/AirSteady/labelimg_dataset" '
            '-m "yolo11s.pt" -s 10 --imgsz 960 --device 0\n'
        )
        print("参数说明：")
        print("  -v / --video        输入视频路径")
        print("  -o / --output-root  输出根目录（脚本会在里面创建 images/ 和 labels/）")
        print("  -m / --model        YOLO 模型路径（默认 yolo11s.pt）")
        print("  -s / --skip         抽帧间隔，默认 10")
        print("  --imgsz             推理分辨率，默认 960")
        print('  --device            设备：GPU 写 "0"、"1"...；CPU 写 "cpu"\n')
        return

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return

    output_root = Path(args.output_root)
    model_path = Path(args.model)

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"

    print("========== 参数确认 ==========")
    print(f"[INFO] 视频路径:      {video_path}")
    print(f"[INFO] 输出根目录:    {output_root}")
    print(f"[INFO] 图片目录:      {images_dir}")
    print(f"[INFO] 标签目录:      {labels_dir}")
    print(f"[INFO] 抽帧间隔 skip: {args.skip}  (每隔 {args.skip} 帧保存一张)")
    print(f"[INFO] 模型路径:      {model_path}")
    print(f"[INFO] 推理分辨率:    {args.imgsz}")
    print(f"[INFO] 设备:          {args.device}")
    print("================================\n")

    image_paths = extract_frames(video_path, images_dir, args.skip)
    if not image_paths:
        print("[WARN] 没有抽取到任何帧，脚本结束。")
        return

    yolo_prelabel(
        image_paths=image_paths,
        labels_dir=labels_dir,
        model_path=model_path,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\n[DONE] 下一步在 LabelImg 中操作：")
    print(f"  1) Open Dir        选择: {images_dir}")
    print(f"  2) Change Save Dir 选择: {labels_dir}")
    print("  3) 左侧格式切换到 YOLO，就可以看到预标注结果进行修标了。")


if __name__ == "__main__":
    main()
