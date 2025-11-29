# version_info.py

"""
统一管理 AirSteady 的版本信息 & 内测代际 ID。
"""

from dataclasses import dataclass


# ============================
# 基本版本信息
# ============================

APP_NAME = "AirSteady"

# 语义版本号
APP_VERSION = "0.3.1"

# 渠道: "trial" / "beta" / "release"
APP_CHANNEL = "trial"

# 构建元信息（例如: 日期 + 流水号 或 Git SHA）
BUILD_META = "2025-11-29-a"

# 授权文件格式版本号（将来改结构时可用）
LICENSE_SCHEMA_VERSION = 1

# 每一波内测的“代际”ID，只要想重新发一次内测，就改这里。
TRIAL_SERIES_ID = "2025Q4_TRIAL2"

# 绝对过期时间 (UTC 秒)，建议通过小脚本算:
#   import datetime
#   dt = datetime.datetime(2026, 1, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
#   print(int(dt.timestamp()))
# Today 1767542400
# 1767542400: 2025-01-05
TRIAL_EXPIRY_EPOCH = 1767542400  # TODO: 换成你需要的时间


@dataclass(frozen=True)
class VersionInfo:
    app_name: str
    version: str
    channel: str
    build_meta: str
    trial_series_id: str

    def full_version_string(self) -> str:
        """
        用于 About 界面 / 日志的完整版本字符串.
        例如: "AirSteady 0.3.1 (trial, build 2025-11-29-a)"
        """
        return f"{self.app_name} {self.version} ({self.channel}, build {self.build_meta})"


def get_version_info() -> VersionInfo:
    return VersionInfo(
        app_name=APP_NAME,
        version=APP_VERSION,
        channel=APP_CHANNEL,
        build_meta=BUILD_META,
        trial_series_id=TRIAL_SERIES_ID,
    )
