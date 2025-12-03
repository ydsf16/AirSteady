#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge two videos with stacking:

- If --mode auto:
    - If video1 is landscape (width >= height) -> vertical stack (top/bottom).
    - Otherwise -> horizontal stack (left/right).
- If --mode vertical:
    - Always stack vertically (video1 on top, video2 on bottom).
- If --mode horizontal:
    - Always stack horizontally (video1 on left, video2 on right).

Additionally:
- The first video region is tagged with a top-right 'Original' label
  whose size scales with resolution.

Usage examples:
    python merge_two_videos.py video1.mp4 video2.mp4 output.mp4
    python merge_two_videos.py video1.mp4 video2.mp4 output.mp4 --mode vertical
"""

import argparse
import os
import sys

import cv2
import numpy as np


def merge_two_videos(
    video_path_1: str,
    video_path_2: str,
    output_path: str,
    mode: str = "auto",  # "auto", "vertical", or "horizontal"
) -> None:
    """Merge two videos by stacking them vertically or horizontally.

    Args:
        video_path_1: Path to first input video.
        video_path_2: Path to second input video.
        output_path: Path to output merged video.
        mode: "auto", "vertical", or "horizontal".
            - "auto": landscape -> vertical, otherwise -> horizontal (based on video1).
            - "vertical": always stack top/bottom (video1 top, video2 bottom).
            - "horizontal": always stack left/right (video1 left, video2 right).
    """
    if mode not in ("auto", "vertical", "horizontal"):
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.isfile(video_path_1):
        raise FileNotFoundError(f"video_path_1 not found: {video_path_1}")
    if not os.path.isfile(video_path_2):
        raise FileNotFoundError(f"video_path_2 not found: {video_path_2}")

    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path_1}")
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path_2}")

    # ---- 基础信息 ----
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(n1, n2)

    if w1 <= 0 or h1 <= 0:
        cap1.release()
        cap2.release()
        raise RuntimeError(f"Invalid size for video1: {w1}x{h1}")
    if w2 <= 0 or h2 <= 0:
        cap1.release()
        cap2.release()
        raise RuntimeError(f"Invalid size for video2: {w2}x{h2}")

    # ---- 选择 FPS：优先用 video1，其次 video2，最后 30 ----
    if fps1 is None or fps1 <= 1e-3 or not np.isfinite(fps1):
        if fps2 is None or fps2 <= 1e-3 or not np.isfinite(fps2):
            fps = 30.0
        else:
            fps = fps2
    else:
        fps = fps1

    # ---- 根据 mode 决定堆叠方式 ----
    if mode == "auto":
        if w1 >= h1:
            stack_mode = "vertical"   # 横屏 -> 上下堆叠
        else:
            stack_mode = "horizontal"  # 竖屏/接近方形 -> 左右堆叠
    else:
        stack_mode = mode

    print(f"[INFO] video1 size: {w1}x{h1}, video2 size: {w2}x{h2}")
    print(f"[INFO] fps: {fps:.3f}, frames: {n1} / {n2} -> using {total_frames}")
    print(f"[INFO] stack_mode: {stack_mode}")

    # ---- 初始基准分辨率：以 video1 为准 ----
    base_w, base_h = w1, h1

    # 先估算不缩放时堆叠后的尺寸
    if stack_mode == "vertical":
        stacked_w = base_w
        stacked_h = base_h * 2
    else:
        stacked_w = base_w * 2
        stacked_h = base_h

    # 限制合并后视频的最大边，避免 4K*2 这种超宽/超高
    MAX_OUT_SIDE = 3840  # 你可以改成 4096 之类
    max_side = max(stacked_w, stacked_h)
    if max_side > MAX_OUT_SIDE:
        scale = MAX_OUT_SIDE / float(max_side)
        base_w = int(round(base_w * scale))
        base_h = int(round(base_h * scale))
        print(f"[INFO] downscale factor for stacking: {scale:.3f}, new base={base_w}x{base_h}")

    # 防止奇数宽高
    if base_w % 2 == 1:
        base_w -= 1
    if base_h % 2 == 1:
        base_h -= 1

    if base_w <= 0 or base_h <= 0:
        cap1.release()
        cap2.release()
        raise RuntimeError(f"Adjusted base size invalid: {base_w}x{base_h}")

    # 计算最终输出分辨率
    if stack_mode == "vertical":
        out_w = base_w
        out_h = base_h * 2
    else:  # horizontal
        out_w = base_w * 2
        out_h = base_h

    print(f"[INFO] final output size: {out_w}x{out_h}")

    # ---- 文本标签配置（给原片打角标，自适应分辨率）----
    raw_label = "Original"
    font = cv2.FONT_HERSHEY_DUPLEX
    base_font_scale = 0.7  # 针对 ~1080 高度的基准字号
    thickness = 2

    # ---- 创建输出目录 ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    print(f"[INFO] output video: {output_path}")

    frame_idx = 0
    try:
        while frame_idx < total_frames:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or frame1 is None or not ret2 or frame2 is None:
                print(f"[WARN] Early EOF at frame {frame_idx}")
                break

            # 确保 BGR 三通道
            if frame1.ndim == 2:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            if frame2.ndim == 2:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

            # 统一到 (base_w, base_h)
            if frame1.shape[1] != base_w or frame1.shape[0] != base_h:
                frame1 = cv2.resize(frame1, (base_w, base_h), interpolation=cv2.INTER_AREA)
            if frame2.shape[1] != base_w or frame2.shape[0] != base_h:
                frame2 = cv2.resize(frame2, (base_w, base_h), interpolation=cv2.INTER_AREA)

            # ---- 在第一个视频区域右上角打 "Original" 角标（自适应字号）----
            h1c, w1c = frame1.shape[:2]
            base_dim = min(w1c, h1c)
            # 以 1080 高度为基准缩放字号
            scale = base_dim / 1080.0
            font_scale = max(0.5, min(1.8, base_font_scale * scale))

            (text_w, text_h), baseline = cv2.getTextSize(
                raw_label, font, font_scale, thickness
            )

            # margin / padding 也随分辨率缩放
            margin = int(max(10, base_dim * 0.02))       # 1080 时约 20px
            pad_x = int(max(6, base_dim * 0.01))         # 水平 padding
            pad_y = int(max(4, base_dim * 0.007))        # 垂直 padding

            x = max(margin, w1c - text_w - margin)       # 右侧
            y = margin + text_h                          # 顶部略往下

            # 半透明黑底 + 轻微阴影
            overlay = frame1.copy()
            bg_tl = (x - pad_x, y - text_h - pad_y)
            bg_br = (x + text_w + pad_x, y + pad_y)
            cv2.rectangle(overlay, bg_tl, bg_br, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame1, 0.6, 0, frame1)

            shadow_color = (0, 0, 0)
            text_color = (255, 255, 255)

            # 阴影
            cv2.putText(
                frame1,
                raw_label,
                (x + 1, y + 1),
                font,
                font_scale,
                shadow_color,
                thickness + 1,
                cv2.LINE_AA,
            )

            # 正文
            cv2.putText(
                frame1,
                raw_label,
                (x, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

            # ---- 堆叠两个画面 ----
            if stack_mode == "vertical":
                # 上下堆叠：frame1 在上，frame2 在下
                combined = np.zeros((out_h, out_w, 3), dtype=frame1.dtype)
                combined[0:base_h, :, :] = frame1
                combined[base_h:base_h * 2, :, :] = frame2
            else:
                # 左右堆叠：frame1 在左，frame2 在右
                combined = np.zeros((out_h, out_w, 3), dtype=frame1.dtype)
                combined[:, 0:base_w, :] = frame1
                combined[:, base_w:base_w * 2, :] = frame2

            writer.write(combined)
            frame_idx += 1

            if frame_idx % 50 == 0 or frame_idx == total_frames:
                print(f"\r[INFO] processed {frame_idx}/{total_frames} frames", end="")

        print(f"\n[INFO] Done. Total written frames: {frame_idx}")
    finally:
        cap1.release()
        cap2.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two videos by stacking (auto / vertical / horizontal), "
                    "and label the first video as 'Original'."
    )
    parser.add_argument("video1", help="Path to the first input video")
    parser.add_argument("video2", help="Path to the second input video")
    parser.add_argument("output", help="Path to the output merged video")
    parser.add_argument(
        "--mode",
        choices=["auto", "vertical", "horizontal"],
        default="auto",
        help=(
            "Stacking mode:\n"
            "  auto       - landscape -> vertical stack, otherwise horizontal (default)\n"
            "  vertical   - always stack top/bottom (video1 on top)\n"
            "  horizontal - always stack left/right (video1 on left)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_two_videos(
        video_path_1=args.video1,
        video_path_2=args.video2,
        output_path=args.output,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
