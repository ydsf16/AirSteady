# core/asset_loader.py
import os
import tempfile
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 和 encrypt_assets.py 中 RAW_KEY 保持完全一致
_SECRET_KEY = b"0123456789ABCDEF0123456789ABCDEF"  # 32 bytes, AES-256 key


def get_default_assets_path() -> str:
    """
    返回加密包路径，假设放在 core/assets/assets.enc。
    """
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    enc_path = base_dir / "assets" / "assets.enc"
    return str(enc_path)


def _decrypt_payload(enc_path: str) -> bytes:
    """
    从 enc_path 读取全部字节，并用 AES-GCM 解密得到 payload。

    encrypt_assets.py 写入格式为：
        [12 字节 nonce][ciphertext_with_tag]

    这里就按同样格式解析：
        nonce = raw[:12]
        cipher = raw[12:]
        payload = AESGCM(key).decrypt(nonce, cipher, None)
    """
    p = Path(enc_path)
    if not p.exists():
        raise FileNotFoundError(f"encrypted model bundle not found: {p}")

    raw = p.read_bytes()
    if len(raw) < 12:
        raise RuntimeError("encrypted bundle too small (len < 12)")

    nonce = raw[:12]
    cipher = raw[12:]

    aesgcm = AESGCM(_SECRET_KEY)
    payload = aesgcm.decrypt(nonce, cipher, associated_data=None)
    return payload


def materialize_assets_to_temp(enc_path: str) -> Tuple[str, str]:
    """
    解密 enc_path，并把 model.pt + tracker.yaml 写到临时目录里。

    payload 格式约定为：
        [4 字节 model_len(big endian)] [model_bytes] [yaml_bytes]

    返回:
        (model_pt_path, tracker_yaml_path)
    """
    print("[asset_loader] materialize_assets_to_temp called with:", enc_path)
    payload = _decrypt_payload(enc_path)

    if len(payload) < 4:
        raise RuntimeError("payload too short")

    model_len = int.from_bytes(payload[:4], "big")
    if model_len <= 0 or model_len > len(payload) - 4:
        raise RuntimeError(
            f"invalid model_len={model_len}, payload_size={len(payload)}"
        )

    model_bytes = payload[4:4 + model_len]
    yaml_bytes = payload[4 + model_len:]

    tmp_dir = Path(tempfile.gettempdir()) / "airsteady_model"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_path = tmp_dir / "model.pt"
    yaml_path = tmp_dir / "tracker.yaml"

    model_path.write_bytes(model_bytes)
    yaml_path.write_bytes(yaml_bytes)

    # print("[asset_loader] model written to:", model_path)
    # print("[asset_loader] tracker yaml written to:", yaml_path)

    return str(model_path), str(yaml_path)


def load_model_and_config(enc_path: str):
    """
    高层封装：
    1. 解密加密包 -> 落盘 model.pt & tracker.yaml
    2. 用 ultralytics.YOLO 正常加载模型
    3. 尝试删除临时的 model.pt（tracker.yaml 保留）
    4. 返回 (model_obj, tracker_cfg_path, device)
    """
    model_path, yaml_path = materialize_assets_to_temp(enc_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"[asset_loader] loading YOLO model from {model_path} on device={device}")

    model = YOLO(model_path)
    model.to(device)

    # 尝试删除临时 model.pt（不影响已经在内存里的模型）
    try:
        Path(model_path).unlink(missing_ok=True)
        # print(f"[asset_loader] deleted temp model file: {model_path}")
    except Exception as e:
        pass
        # 删除失败就算了，不影响正常使用
        # print(f"[asset_loader] failed to delete temp model file: {e}")

    return model, yaml_path, device
