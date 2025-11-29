# encrypt_assets.py
import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_assets(model_path: str, yaml_path: str, out_path: str, key: bytes) -> None:
    """
    把 model.pt + tracker.yaml 打包并用 AES-GCM 加密，输出到 out_path。
    打包格式：
      [4 字节大端 model_len][model_bytes][yaml_bytes]
    """
    assert len(key) == 32, "key 必须是 32 字节 (AES-256)"

    model_path = Path(model_path)
    yaml_path = Path(yaml_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as f:
        model_bytes = f.read()
    with yaml_path.open("rb") as f:
        yaml_bytes = f.read()

    model_len = len(model_bytes).to_bytes(4, "big")
    payload = model_len + model_bytes + yaml_bytes

    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # GCM 常用 12 字节 nonce
    cipher = aesgcm.encrypt(nonce, payload, associated_data=None)

    with out_path.open("wb") as f:
        f.write(nonce + cipher)

    print(f"OK, encrypted to {out_path}")


if __name__ == "__main__":
    # ============================
    # 这里是“明文密钥”：只在开发机用
    # ============================
    # 建议你自己随便造一串 32 字节字符串替换掉下面这行
    RAW_KEY = b"0123456789ABCDEF0123456789ABCDEF"  # 示例（32B）

    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "model" / "model.pt"
    yaml_path = base_dir / "model" / "tracker.yaml"
    out_path = base_dir / "assets" / "assets.enc"

    encrypt_assets(str(model_path), str(yaml_path), str(out_path), RAW_KEY)
