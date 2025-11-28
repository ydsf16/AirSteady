import os
import sys
import tempfile
from uuid import uuid4
from pathlib import Path
import logging


def get_airsteady_cache_dir() -> str:
    """
    返回 AirSteady 的缓存目录路径（按平台放到系统推荐的 cache 位置）。
    - Windows:  %LOCALAPPDATA%\AirSteady\cache
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


def _resource_root() -> str:
    """
    返回打包后运行时的资源根目录：
    - 裸跑：就是当前文件所在目录；
    - PyInstaller：使用 sys._MEIPASS；
    - Nuitka：使用 sys.argv[0] 所在目录。
    """
    # PyInstaller onefile/onedir
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS

    # Nuitka 或其他 “frozen” 运行方式
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))

    # 普通 Python 环境
    return os.path.dirname(os.path.abspath(__file__))


def get_ffmpeg_path() -> str:
    """
    返回 ffmpeg 可执行文件的绝对路径。
    优先级：
    1. 环境变量 AIRSTEADY_FFMPEG
    2. 程序目录下的内置 ffmpeg（比如：bin/ffmpeg.exe）
    3. 退而求其次：'ffmpeg'（依赖系统 PATH）
    """
    # 1) 环境变量优先（便于你自己调试）
    env_path = os.environ.get("AIRSTEADY_FFMPEG")
    if env_path and os.path.exists(env_path):
        return env_path

    root = _resource_root()

    # 2) 约定一个固定的相对路径，比如：<root>/bin/ffmpeg(.exe)
    #    你打包时只要把 ffmpeg 放到这个位置就可以了。
    candidates = []
    if os.name == "nt":
        candidates.append(os.path.join(root, "bin", "ffmpeg.exe"))
        candidates.append(os.path.join(root, "ffmpeg.exe"))
    else:
        candidates.append(os.path.join(root, "bin", "ffmpeg"))
        candidates.append(os.path.join(root, "ffmpeg"))

    for c in candidates:
        if os.path.exists(c):
            return c

    # 3) 最后兜底：假设系统 PATH 有 ffmpeg（开发机上一般有）
    return "ffmpeg"


# ====================== 日志相关工具 ======================

def get_airsteady_log_dir() -> str:
    """
    返回 AirSteady 的缓存目录路径（按平台放到系统推荐的 logs 位置）。
    - Windows:  %LOCALAPPDATA%\AirSteady\logs
    - macOS:    ~/Library/Logss/AirSteady
    - Linux:    ~/.logs/airsteady 或 $XDG_LOGS_HOME/airsteady
    """
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA", tempfile.gettempdir())
        logs_dir = os.path.join(base, "AirSteady", "logs")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logss")
        logs_dir = os.path.join(base, "AirSteady")
    else:
        # Linux / Unix
        base = os.getenv("XDG_LOGS_HOME", os.path.expanduser("~/.logs"))
        logs_dir = os.path.join(base, "airsteady")

    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def init_logging(logger_name: str = "airsteady") -> logging.Logger:
    """
    初始化 AirSteady 的日志系统：
    - 每次启动都覆盖同一个 log 文件（只保留本次运行的日志）；
    - 日志会写入文件，方便“问题反馈”里打包；
    - 不改变现有 print 行为（print 仍然按原样输出到控制台）。

    调用多次是安全的（只会初始化一次）。
    """
    logger = logging.getLogger(logger_name)

    # 已经初始化过就复用
    if getattr(logger, "_airsteady_inited", False):
        return logger

    logger.setLevel(logging.INFO)

    log_dir = get_airsteady_log_dir()
    log_path = os.path.join(log_dir, "airsteady.log")

    # 每次运行覆盖写入，保证“循环记录，只保留当次”
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 标记一下，避免重复配置
    logger._airsteady_inited = True  # type: ignore[attr-defined]

    return logger
