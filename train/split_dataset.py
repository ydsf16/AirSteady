#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从预标注数据集随机划分 train / val 集合。

输入目录结构（--input）假定为：
    input_root/
      images/
        *.jpg / *.png ...
      labels/
        *.txt
        [可选] classes.txt   # 如果存在，会被拷贝到输出 labels 根目录

输出目录结构（--output）将生成：
    output_root/
      images/
        train/
        val/
      labels/
        train/
        val/

用法示例：

    python split_dataset.py ^
      --input E:/AirSteady/code/AirSteady/train/datasets/test ^
      --output E:/AirSteady/code/AirSteady/train/datasets/test_split ^
      --val-ratio 0.2 ^
      --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def list_images(images_dir: Path) -> List[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    paths: List[Path] = []
    for ext in exts:
        paths.extend(images_dir.glob(f"*{ext}"))
    # 去重并排序（稳一点）
    paths = sorted(set(paths))
    return paths


def copy_pair(
    img_path: Path,
    src_labels: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
):
    stem = img_path.stem
    label_path = src_labels / f"{stem}.txt"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, dst_img_dir / img_path.name)
    if label_path.exists():
        shutil.copy2(label_path, dst_lbl_dir / label_path.name)
    else:
        # 如果没有对应标签，创建一个空的 txt（YOLO 训练可以接受）
        (dst_lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")


def split_dataset(
    input_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    src_images = input_root / "images"
    src_labels = input_root / "labels"

    if not src_images.is_dir():
        raise RuntimeError(f"[ERROR] 输入目录缺少 images/ 子目录: {src_images}")
    if not src_labels.is_dir():
        raise RuntimeError(f"[ERROR] 输入目录缺少 labels/ 子目录: {src_labels}")

    # 输出目录结构
    dst_img_train = output_root / "images" / "train"
    dst_img_val = output_root / "images" / "val"
    dst_lbl_train = output_root / "labels" / "train"
    dst_lbl_val = output_root / "labels" / "val"

    for d in [dst_img_train, dst_img_val, dst_lbl_train, dst_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 列出所有图片
    image_paths = list_images(src_images)
    n = len(image_paths)
    if n == 0:
        raise RuntimeError(f"[ERROR] No images found in {src_images}")

    # 随机划分
    random.seed(seed)
    random.shuffle(image_paths)

    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    train_imgs = image_paths[:n_train]
    val_imgs = image_paths[n_train:]

    print(f"[INFO] Total images: {n}")
    print(f"[INFO] Train: {len(train_imgs)}, Val: {len(val_imgs)} "
          f"(val_ratio={val_ratio}, seed={seed})")

    # 复制 train
    for img in train_imgs:
        copy_pair(img, src_labels, dst_img_train, dst_lbl_train)

    # 复制 val
    for img in val_imgs:
        copy_pair(img, src_labels, dst_img_val, dst_lbl_val)

    # 如果有 classes.txt，一并拷贝到 output_root/labels 下（方便 LabelImg / 记录）
    classes_src = src_labels / "classes.txt"
    if classes_src.exists():
        classes_dst_root = output_root / "labels"
        classes_dst_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_src, classes_dst_root / "classes.txt")
        print(f"[INFO] Copied classes.txt -> {classes_dst_root / 'classes.txt'}")

    print(f"[DONE] Dataset created at: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机划分 YOLO 数据集为 train/val（从 images/ + labels/ 输入）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入数据根目录（包含 images/ 和 labels/），例如：E:/.../datasets/test",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出数据根目录，将在其中生成 images/train,val 和 labels/train,val",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例，例如 0.2 表示 20%% 数据用于 val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证多次运行划分一致）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_root}")
        return

    print("========== 参数确认 ==========")
    print(f"[INFO] 输入目录:   {input_root}")
    print(f"[INFO] 输出目录:   {output_root}")
    print(f"[INFO] val_ratio:  {args.val_ratio}")
    print(f"[INFO] seed:       {args.seed}")
    print("================================\n")

    split_dataset(
        input_root=input_root,
        output_root=output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
