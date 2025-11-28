import os
import sys
import tempfile
from uuid import uuid4

def get_airsteady_cache_dir() -> str:
    """
    返回 AirSteady 的缓存目录路径（按平台放到系统推荐的 cache 位置）。
    - Windows:  %LOCALAPPDATA%\\AirSteady\\cache
    - macOS:    ~/Library/Caches/AirSteady
    - Linux:    ~/.cache/airsteady 或 $XDG_CACHE_HOME/airsteady
    """
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA", tempfile.gettempdir())
        cache_dir = os.path.join(base, "AirSteady", "cache")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
        cache_dir = os.path.join(base, "AirSteady")
    else:
        # Linux / Unix
        base = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base, "airsteady")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def make_temp_video_path(suffix: str = ".mp4") -> str:
    """
    生成一个临时视频文件路径，如：
    <cache_dir>/track_tmp_<uuid>.mp4
    """
    cache_dir = get_airsteady_cache_dir()
    filename = f"track_tmp_{uuid4().hex}{suffix}"
    return os.path.join(cache_dir, filename)
