import cv2
import numpy as np
import os
import argparse
from collections import defaultdict
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="AirSteady · 航迹稳拍 — YOLO 跟踪 + 稳像拼图 Demo"
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="输入视频路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="输出视频路径（不指定则在输入视频旁生成 *_airsteady_concat.mp4）"
    )
    parser.add_argument(
        "-c", "--crop_ratio",
        type=float,
        default=0.8,
        help="裁剪窗口占原始宽/高的比例 (0~1)，默认 0.6"
    )
    parser.add_argument(
        "-e", "--ema_alpha",
        type=float,
        default=0.9,
        help="EMA 平滑系数 (0~1)，越小越稳，默认 0.2"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    crop_ratio = float(args.crop_ratio)
    ema_alpha = float(args.ema_alpha)

    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError(f"crop_ratio 必须在 (0, 1]，当前为 {crop_ratio}")
    if not (0.0 < ema_alpha <= 1.0):
        raise ValueError(f"ema_alpha 必须在 (0, 1]，当前为 {ema_alpha}")

    # 自动生成输出路径
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(video_path)
        output_path = base + "_airsteady_concat.mp4"

    print(f"输入视频: {video_path}")
    print(f"输出视频: {output_path}")
    print(f"crop_ratio = {crop_ratio}, ema_alpha = {ema_alpha}")

    # ======================
    # 1. 加载 YOLO 模型
    # ======================
    model = YOLO("yolo11n.pt")

    # 候选类别：airplane / bird / kite
    names = model.names  # dict: {0: 'person', 1: 'bicycle', ...}
    candidate_names = ["airplane", "bird", "kite"]
    candidate_cls_ids = [i for i, n in names.items() if n in candidate_names]

    if not candidate_cls_ids:
        raise RuntimeError(
            f"模型里没有 {candidate_names} 这些类别，检查权重是否是 COCO 预训练。"
        )

    print("候选类别:", {i: names[i] for i in candidate_cls_ids})

    # ======================
    # 2. 打开视频
    # ======================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # 兜底

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("视频为空。")

    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到开头

    # 判断横屏还是竖屏，决定拼接方向
    is_landscape = w >= h
    if is_landscape:
        # 横屏视频 → 上下拼接（竖着拼）
        out_width = w
        out_height = 2 * h
        concat_mode = "vertical"
    else:
        # 竖屏视频 → 左右拼接（横着拼）
        out_width = 2 * w
        out_height = h
        concat_mode = "horizontal"

    print(f"视频尺寸: {w}x{h}, 模式: {concat_mode}, 输出尺寸: {out_width}x{out_height}")

    # ======================
    # 3. 状态变量
    # ======================
    track_history = defaultdict(lambda: [])
    target_track_id = None          # 当前锁定的目标 track id
    smoothed_center = None          # EMA 平滑后的中心

    # 裁剪窗口尺寸
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    # ======================
    # 4. VideoWriter
    # ======================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    def crop_stabilized(frame, center):
        """根据平滑后的中心点裁剪 + resize 回原尺寸"""
        cx, cy = center
        # 以 center 为中心算窗口左上角
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)

        # 防止越界
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]

        # 拉回原图大小，方便对比 / 拼接
        stabilized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return stabilized

    # ======================
    # 5. 主循环
    # ======================
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO 跟踪（persist=True 保留 track id）
        result = model.track(frame, persist=True, verbose=False)[0]

        # 带框的可视化帧
        vis_frame = result.plot()

        candidate_boxes = []
        candidate_track_ids = []

        # 取出 tracking 结果
        if result.boxes is not None and result.boxes.id is not None:
            boxes_xywh = result.boxes.xywh.cpu().numpy()  # [x, y, w, h] 中心坐标
            cls_ids = result.boxes.cls.int().cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            # 收集 airplane / bird / kite 这些候选
            for (x, y, w_box, h_box), cls_id, tid in zip(boxes_xywh, cls_ids, track_ids):
                if cls_id in candidate_cls_ids:
                    candidate_boxes.append((x, y, w_box, h_box))
                    candidate_track_ids.append(tid)

            # 画所有 track 的小轨迹（只是好看，可注释掉）
            for (x, y, w_box, h_box), tid in zip(boxes_xywh, track_ids):
                track = track_history[tid]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                pts = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    vis_frame, [pts],
                    isClosed=False,
                    color=(230, 230, 230),
                    thickness=2
                )

        # ========= 选择要跟踪的那一只/架 =========
        current_center = None

        if candidate_boxes:
            # 如果还没锁定目标，选一个（例如面积最大）
            if target_track_id is None:
                areas = [w_box * h_box for (_, _, w_box, h_box) in candidate_boxes]
                best_idx = int(np.argmax(areas))
                target_track_id = candidate_track_ids[best_idx]

            # 在候选里找到当前锁定 track_id 的框
            for (x, y, w_box, h_box), tid in zip(candidate_boxes, candidate_track_ids):
                if tid == target_track_id:
                    current_center = np.array([x, y], dtype=np.float32)
                    # 画出当前锁定目标中心（绿色点）
                    cv2.circle(vis_frame, (int(x), int(y)), 6, (0, 255, 0), -1)
                    break

            # 当前帧找不到那只/那架：视作丢失，下次重选
            if current_center is None:
                target_track_id = None

        # ========= EMA 平滑中心 =========
        if current_center is not None:
            if smoothed_center is None:
                smoothed_center = current_center.copy()
            else:
                smoothed_center = (1 - ema_alpha) * smoothed_center + ema_alpha * current_center

        # ========= 生成稳定画面 + 打水印 =========
        if smoothed_center is not None:
            stabilized_frame = crop_stabilized(frame, smoothed_center)

            # 打水印
            text = "AirSteady"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            # 右上角偏一点
            x_text = w - text_w - 20
            y_text = 30

            # 黑底块（简单版）
            cv2.rectangle(
                stabilized_frame,
                (x_text - 10, y_text - text_h - 5),
                (x_text + text_w + 10, y_text + 5),
                (0, 0, 0),
                thickness=-1
            )
            cv2.putText(
                stabilized_frame,
                text,
                (x_text, y_text),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )
        else:
            # 一开始还没有中心时，就让 stabilized_frame = 原图（避免 None）
            stabilized_frame = frame.copy()

        # ========= 拼接 + 显示 + 写文件 =========
        vis_frame_resized = cv2.resize(vis_frame, (w, h))
        stabilized_frame_resized = cv2.resize(stabilized_frame, (w, h))

        if concat_mode == "horizontal":
            # 竖屏视频 → 左右拼
            concat_frame = np.hstack((vis_frame_resized, stabilized_frame_resized))
        else:
            # 横屏视频 → 上下拼
            concat_frame = np.vstack((vis_frame_resized, stabilized_frame_resized))

        cv2.imshow("AirSteady - Original | Stabilized", concat_frame)
        out.write(concat_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # ======================
    # 6. 收尾
    # ======================
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done. 保存到:", output_path)


if __name__ == "__main__":
    main()
