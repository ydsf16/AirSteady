# license_guard.py
import json
import os
import time
import socket
import hashlib
import urllib.request
import urllib.error
import email.utils
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils import get_airsteady_log_dir
from version_info import (
    APP_CHANNEL,
    TRIAL_EXPIRY_EPOCH,
    TRIAL_SERIES_ID,
    LICENSE_SCHEMA_VERSION,
)


# ============================
# 1) 配置 & 常量
# ============================

# 允许的本地时间回拨范围（秒），例如 3 天
ALLOWED_CLOCK_DRIFT = 3 * 24 * 3600

# 一个硬编码盐，用于签名。随便写一长串难猜字符串即可。
_SECRET_SALT = "AirSteady_Internal_Trial_v0.3_with_net_time_and_series_2389fj23_!@#"

# UI 展示用的 HTML 文案（可按需调整）
EXPIRED_MESSAGE_HTML = """
<b>当前内测版本已过期。</b><br><br>
请联系作者获取正式版或新的内测版本：<br>
微信：YDSF16<br>
官网：<a href="https://ai.feishu.cn/wiki/VVsDwawHxiaOIlkvOFMce7fvnhe">https://ai.feishu.cn/wiki/VVsDwawHxiaOIlkvOFMce7fvnhe</a><br>
"""

CLOCK_TAMPERED_MESSAGE_HTML = """
<b>检测到系统时间异常。</b><br><br>
请检查系统时间是否被大幅回拨。<br>
如有疑问，请联系作者。
"""


class LicenseExpiredError(Exception):
    """版本已过期."""
    pass


class ClockTamperedError(Exception):
    """检测到系统时间异常/回拨."""
    pass


@dataclass
class LicenseState:
    max_time_seen: float
    expired: bool
    trial_series_id: str
    schema_version: int


# ============================
# 2) 本地状态文件
# ============================

def _get_sys_dir() -> Path:
    """
    使用 AirSteady 的 logs 目录，下面挂一个 .sys 作为内部状态目录。
    例如:
      Windows: %LOCALAPPDATA%\\AirSteady\\logs\\.sys
      macOS :  ~/Library/Logss/AirSteady/.sys
      Linux :  ~/.logs/airsteady/.sys
    """
    base = Path(get_airsteady_log_dir())
    sys_dir = base / ".sys"
    sys_dir.mkdir(parents=True, exist_ok=True)
    return sys_dir


def _get_license_file() -> Path:
    return _get_sys_dir() / "license_state.json"


def _compute_checksum(max_time_seen: float, expired: bool, trial_series_id: str) -> str:
    """
    使用 max_time_seen + expired + hostname + trial_series_id + secret salt 做签名。
    手动修改文件通常会导致校验失败。
    """
    host = socket.gethostname()
    raw = f"{max_time_seen:.3f}|{int(expired)}|{trial_series_id}|{host}|{_SECRET_SALT}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_state() -> Optional[LicenseState]:
    path = _get_license_file()
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # 文件损坏/不是合法 JSON，当作不存在
        return None

    try:
        max_time_seen = float(data.get("max_time_seen", 0.0))
        expired = bool(data.get("expired", False))
        checksum = str(data.get("checksum", ""))
        trial_series_id = str(data.get("trial_series_id", "UNKNOWN_SERIES"))
        schema_version = int(data.get("schema_version", 0))
    except Exception:
        return None

    # 校验签名
    expected = _compute_checksum(max_time_seen, expired, trial_series_id)
    if checksum != expected:
        # 说明被人手动改过 / 被破坏，直接按「已过期」处理
        raise LicenseExpiredError("License file tampered")

    return LicenseState(
        max_time_seen=max_time_seen,
        expired=expired,
        trial_series_id=trial_series_id,
        schema_version=schema_version,
    )


def _save_state(state: LicenseState) -> None:
    path = _get_license_file()
    checksum = _compute_checksum(state.max_time_seen, state.expired, state.trial_series_id)
    data = {
        "max_time_seen": state.max_time_seen,
        "expired": state.expired,
        "checksum": checksum,
        "trial_series_id": state.trial_series_id,
        "schema_version": state.schema_version,
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def _get_now_local_epoch() -> float:
    # 统一用 time.time()，避免时区问题
    return float(time.time())


# ============================
# 3) 网络时间获取
# ============================

def _fetch_http_date(url: str, timeout: float = 1.0) -> Optional[float]:
    """
    通过 HTTP HEAD 请求拿服务器的 Date 头。
    返回 epoch 秒，失败返回 None。
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            date_str = resp.headers.get("Date")
            if not date_str:
                return None
            dt = email.utils.parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
            return float(dt.timestamp())
    except Exception:
        return None


def get_network_time_epoch() -> Optional[float]:
    """
    尝试从几个公共网站获取 HTTP Date 作为网络时间。
    成功则返回 epoch 秒，全部失败返回 None（离线 / 被拦截）。
    """
    urls = [
        "https://www.baidu.com",
        "https://www.bing.com",
        "https://www.microsoft.com",
    ]
    for url in urls:
        ts = _fetch_http_date(url)
        if ts is not None:
            return ts
    return None


# ============================
# 4) 核心检查逻辑（含版本代际）
# ============================

def check_time_limit() -> None:
    """
    核心检查逻辑（带网络时间 + 代际版本）:

    - 若 APP_CHANNEL == "release"，直接返回（可视需求改成仅做轻度防回拨）；
    - 优先使用网络时间作为“真时间”；否则退化为本地时间；
    - 从本地读取 LicenseState:
        * 若 trial_series_id 与当前 TRIAL_SERIES_ID 不同，
          说明这是新一代内测，重新初始化状态；
        * 若 expired=True，直接抛 LicenseExpiredError；
        * 若时间被回拨太多，抛 ClockTamperedError；
        * 若当前时间超过 TRIAL_EXPIRY_EPOCH，写入 expired=True 并抛 LicenseExpiredError；
        * 其他情况更新 max_time_seen 并正常返回。
    """
    # 正式版直接跳过（如果你希望正式版也防回拨，可以在这里改逻辑）
    if APP_CHANNEL == "release":
        return

    now_local = _get_now_local_epoch()
    net_ts = get_network_time_epoch()

    if net_ts is not None:
        # 有网络时间时，作为主要参考
        ref_now = max(now_local, net_ts)
    else:
        # 无网络时退化为本地时间方案
        ref_now = now_local

    state = _load_state()

    # 4.1 第一次运行 或 文件不存在：初始化
    if state is None:
        state = LicenseState(
            max_time_seen=ref_now,
            expired=False,
            trial_series_id=TRIAL_SERIES_ID,
            schema_version=LICENSE_SCHEMA_VERSION,
        )
        _save_state(state)
        if ref_now > TRIAL_EXPIRY_EPOCH:
            raise LicenseExpiredError("This trial build has expired.")
        return

    # 4.2 代际升级：旧系列 -> 新系列（例如 2025Q4_TRIAL1 -> 2026Q1_TRIAL1）
    if state.trial_series_id != TRIAL_SERIES_ID:
        # 说明这是新一代内测，重置状态给一次新试用期
        state = LicenseState(
            max_time_seen=ref_now,
            expired=False,
            trial_series_id=TRIAL_SERIES_ID,
            schema_version=LICENSE_SCHEMA_VERSION,
        )
        _save_state(state)
        if ref_now > TRIAL_EXPIRY_EPOCH:
            # 新一代内测的时间本身已经过了
            raise LicenseExpiredError("This new trial build has already expired.")
        return

    # 4.3 已过期标记 → 永久过期（同一代内）
    if state.expired:
        raise LicenseExpiredError("This trial build has expired (flag).")

    # 4.4 检查时间回拨
    if ref_now + ALLOWED_CLOCK_DRIFT < state.max_time_seen:
        raise ClockTamperedError("System clock seems to have been set back.")

    # 正常情况：更新 max_time_seen
    if ref_now > state.max_time_seen:
        state.max_time_seen = ref_now

    # 4.5 绝对过期时间判断
    if ref_now > TRIAL_EXPIRY_EPOCH:
        state.expired = True
        _save_state(state)
        raise LicenseExpiredError("This trial build has expired (time).")

    # 4.6 正常：只更新状态后返回
    _save_state(state)
