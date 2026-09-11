# 整合的IMU分析工具箱（huace AHRS 二进制协议支持版）
# 分析方法严格继承自 IMU_Analysis_yuanshen.py 参考代码
# 协议字段定义依据 huace_data/AHRS.json (schema_version 4)
# 与《IMU通信协议方案.docx》1.3 节核对一致
import os, sys, re, struct, datetime, textwrap, zlib

# ---- stdout/stderr UTF-8 强制编码（修复 Windows PowerShell GBK 编码报错）----
# PowerShell 默认 stdout 编码为 GBK/cp936，当 print() 输出含 '²°σμ√' 等特殊字符
# 时会抛 UnicodeEncodeError 中断程序流（这会让 run_full_analysis 部分步骤被跳过）。
# 这里在最早的位置强制把 stdout/stderr 改为 UTF-8，避免后续 print 中断。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, Exception):
    # 某些环境下没有 reconfigure 方法，回退到 io.TextIOWrapper 包装
    import io as _io
    try:
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # 极端情况下保持原样

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.font_manager as fm
import chardet
import allantools as at
from scipy.signal import welch
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# ---- 中文字体支持 ----
# 按优先级排序的中文字体关键字（避免选到 SimSun-ExtG 这类只含扩展 CJK 的字体）
_CN_FONT_PRIORITY = [
    'microsoft yahei',   # 微软雅黑 - Windows 首选
    'simhei',            # 黑体 - Windows 备选
    'simsun',            # 宋体（注意：会排除 SimSun-ExtB/ExtG 等扩展子字体）
    'msyh',              # msyh.ttc 文件名
    'noto sans cjk sc', # Noto Sans CJK SC
    'noto sans sc',      # Noto Sans SC
    'source han sans sc',# 思源黑体
    'wenquanyi',         # 文泉驿
    'pingfang sc',       # 苹方 - macOS
    'heiti sc',          # 黑体-简 - macOS
]

# 收集所有可用中文字体，按优先级排序后取第一个
_CN_FONT = None
_CN_FONT_FALLBACKS = []   # 备用字体列表（用于 sans-serif 回退）
_available_fonts_map = {}  # name -> FontProperties
for _fname in fm.findSystemFonts():
    try:
        _fp = fm.FontProperties(fname=_fname)
        _fn = _fp.get_name()
        _fn_lower = _fn.lower()
        # 排除明显只含扩展 CJK 的子字体（如 SimSun-ExtB/ExtG）
        if 'ext' in _fn_lower and 'simsun' in _fn_lower:
            continue
        for _kw in _CN_FONT_PRIORITY:
            if _kw in _fn_lower and _kw not in _available_fonts_map:
                _available_fonts_map[_kw] = _fp
                break
    except Exception:
        continue

# 按优先级顺序选第一个可用的作为主字体
for _kw in _CN_FONT_PRIORITY:
    if _kw in _available_fonts_map:
        _CN_FONT = _available_fonts_map[_kw]
        break

# 构造备选字体名列表（用于 matplotlib 的 sans-serif 回退链）
_CN_FALLBACK_NAMES = []
for _kw in _CN_FONT_PRIORITY:
    if _kw in _available_fonts_map:
        _name = _available_fonts_map[_kw].get_name()
        if _name not in _CN_FALLBACK_NAMES:
            _CN_FALLBACK_NAMES.append(_name)

# 设置 matplotlib 字体配置
if _CN_FONT is not None:
    _main_name = _CN_FONT.get_name()
    print(f"[字体] 主中文字体: {_main_name} ({_CN_FONT.get_file()})")
    print(f"[字体] 备选回退链: {_CN_FALLBACK_NAMES}")
    # 设置 font.family = sans-serif，并把中文字体放到 sans-serif 列表前面
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # 备选链 + DejaVu Sans 兜底（英文/数字/数学符号）
    matplotlib.rcParams['font.sans-serif'] = _CN_FALLBACK_NAMES + ['DejaVu Sans']
else:
    print("[字体] 警告：未检测到任何中文字体，中文将显示为豆腐块")
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================================================================
# huace AHRS 二进制协议常量 (54 字节/帧)
# 依据 huace_data/AHRS.json 定义（little endian，无校验和定义）：
#   0-1:  报文头 0xA5 0xD5
#   2-3:  id     (uint16, 消息ID；实测全文件恒为 2)
#   4-5:  length (uint16, 数据长度；实测全文件恒为 44)
#   6-9:  时间   (uint32, 单位 ms，设备时间戳)
#   10-13: Pitch  (float32, °)
#   14-17: Roll   (float32, °)
#   18-21: Yaw    (float32, °)
#   22-25: accx   (float32, m/s²)
#   26-29: accy   (float32, m/s²)
#   30-33: accz   (float32, m/s²)
#   34-37: gyox   (float32, °/s)
#   38-41: gyoy   (float32, °/s)
#   42-45: gyoz   (float32, °/s)
#   46-49: temper (float32, °C)
#   50-53: crc    (uint32；AHRS.json 中 checksum 为 "none"，
#                  算法未定义，解析时不做校验)
# 注：单位 m/s² / °/s / °C 已由用户确认（2026-09-07），非猜测。
# ===================================================================
FRAME_SIZE = 54
FRAME_SYNC_BYTES = b"\xA5\xD5"      # 同步字 0xA5 0xD5
FRAME_SYNC_OFFSET = 0
FRAME_ID_OFFSET = 2                  # 消息ID (uint16 LE)
FRAME_LEN_OFFSET = 4                 # 数据长度 (uint16 LE)
FRAME_TS_OFFSET = 6                  # 设备时间戳 (uint32 LE, ms)
FRAME_PITCH_OFFSET = 10              # Pitch (float32 LE, °)
FRAME_ROLL_OFFSET = 14               # Roll (float32 LE, °)
FRAME_YAW_OFFSET = 18                # Yaw (float32 LE, °)
FRAME_ACC_OFFSET = 22                # 加速度计 X/Y/Z (3×float32 LE, m/s²)
FRAME_GYRO_OFFSET = 34               # 陀螺仪 X/Y/Z (3×float32 LE, °/s)
FRAME_TEMP_OFFSET = 46               # 温度 (float32 LE, °C)
FRAME_CRC_OFFSET = 50                # crc (uint32 LE，算法未定义，不校验)
# 设备时间戳为 uint32 毫秒计数，模 2^32（约 49.7 天回绕一次）
TIMESTAMP_MODULUS = 2 ** 32
# 帧计数字段不存在：id 为消息ID（恒定），缺帧审计改用设备时间戳。
# 以下常量仅保留给 CSV 路径（若 CSV 含 frameCount 列时的审计口径，继承自华依版）
FRAME_COUNTER_MIN = 1
FRAME_COUNTER_MAX = 60000
FRAME_COUNTER_MODULUS = FRAME_COUNTER_MAX


# Allan 显示网格的公共入口。实际实现委托给分析器中的静态函数，
# 便于两种设备使用同一测试接口，同时不触碰各自的数据解析结构。
def build_allan_display_taus(n_samples, sample_rate, points_per_decade=15):
    return IMUDataAnalyzer._build_allan_display_tau_grid(
        n_samples, sample_rate, points_per_decade
    )


def classify_allan_support(n_terms):
    """按显示用支持项门槛分类；只返回样式标签，不修改曲线。"""
    ns = np.asarray(n_terms, dtype=float)
    support = np.full(ns.shape, "low_n_terms_lt_5", dtype=object)
    support[np.isfinite(ns) & (ns >= 20)] = "normal_n_terms_ge_20"
    support[np.isfinite(ns) & (ns >= 5) & (ns < 20)] = (
        "limited_n_terms_5_to_19"
    )
    return support

# 缩放系数：AHRS.json 中所有通道 scale=1.0，数据为 IEEE float32 直接物理量
SCALE_ACC = 1.0        # m/s²
SCALE_GYRO = 1.0       # °/s
SCALE_EULER = 1.0      # °
SCALE_TEMP = 1.0       # °C

# 帧解析结构体
_FMT_H = struct.Struct('<H')
_FMT_I = struct.Struct('<I')
_FMT_EULER = struct.Struct('<3f')
_FMT_SENSOR = struct.Struct('<3f')
_FMT_F = struct.Struct('<f')


def _frame_crc32_ok(buf: bytes) -> bool:
    """CRC32 校验。

    依据：协议文档 1.3 节帧格式 `A5 D5 | id:u16 | length:u16 | payload | crc32:u32`；
    文档未定义 CRC 参数，算法由实测确定（2026-09-08，对 1,790,738 帧全量
    匹配率 100%）：zlib CRC-32 (ISO-HDLC) 覆盖 frame[2:50]（id+length+payload，
    不含帧头 A5 D5 与 crc 字段自身），小端存储于帧尾 4 字节。
    """
    if len(buf) < FRAME_SIZE:
        return False
    crc_file, = _FMT_I.unpack_from(buf, FRAME_CRC_OFFSET)
    return (zlib.crc32(buf[2:FRAME_CRC_OFFSET]) & 0xFFFFFFFF) == crc_file


def _parse_one_frame(buf: bytes, verify_checksum: bool = True) -> dict | None:
    """解析单帧 huace AHRS 54字节数据

    verify_checksum=True 时执行 CRC32 校验（算法见 _frame_crc32_ok），
    校验失败返回 None（由调用方计入 CRC 拒绝统计）。
    """
    if len(buf) < FRAME_SIZE:
        return None
    if buf[0:2] != FRAME_SYNC_BYTES:
        return None

    try:
        msg_id = _FMT_H.unpack_from(buf, FRAME_ID_OFFSET)[0]
        data_len = _FMT_H.unpack_from(buf, FRAME_LEN_OFFSET)[0]
        ts_ms = _FMT_I.unpack_from(buf, FRAME_TS_OFFSET)[0]
        pitch, roll, yaw = _FMT_EULER.unpack_from(buf, FRAME_PITCH_OFFSET)
        acc_raw = _FMT_SENSOR.unpack_from(buf, FRAME_ACC_OFFSET)
        gyro_raw = _FMT_SENSOR.unpack_from(buf, FRAME_GYRO_OFFSET)
        temp_raw = _FMT_F.unpack_from(buf, FRAME_TEMP_OFFSET)[0]
    except Exception:
        return None

    if verify_checksum and not _frame_crc32_ok(buf):
        return None

    return {
        "msg_id": msg_id,
        "data_len": data_len,
        "timestamp_ms": ts_ms,
        "temp": temp_raw * SCALE_TEMP,
        "acc_x": acc_raw[0] * SCALE_ACC,
        "acc_y": acc_raw[1] * SCALE_ACC,
        "acc_z": acc_raw[2] * SCALE_ACC,
        "gyro_x": gyro_raw[0] * SCALE_GYRO,
        "gyro_y": gyro_raw[1] * SCALE_GYRO,
        "gyro_z": gyro_raw[2] * SCALE_GYRO,
        "pitch": float(pitch * SCALE_EULER),
        "roll": float(roll * SCALE_EULER),
        "yaw": float(yaw * SCALE_EULER),
    }


def parse_binary_file(filepath: str, max_frames: int = 0,
                      verify_checksum: bool = True) -> pd.DataFrame:
    """解析整个二进制文件，返回 DataFrame 及固定帧槽质量属性。"""
    with open(filepath, "rb") as fh:
        raw = fh.read()

    file_size = len(raw)
    n_frames_total = file_size // FRAME_SIZE
    if max_frames > 0:
        n_frames_total = min(n_frames_total, max_frames)

    rows = []
    valid_count = 0
    bad_sync_count = 0
    rejected_count = 0
    crc_rejected_count = 0
    offset = 0
    for i in range(n_frames_total):
        buf = raw[offset:offset + FRAME_SIZE]
        if len(buf) < FRAME_SIZE:
            break
        if buf[FRAME_SYNC_OFFSET:FRAME_SYNC_OFFSET + 2] != FRAME_SYNC_BYTES:
            bad_sync_count += 1
            offset += FRAME_SIZE
            continue
        frame = _parse_one_frame(buf, verify_checksum=verify_checksum)
        if frame is not None:
            valid_count += 1
            rows.append(frame)
        else:
            rejected_count += 1
            # 区分拒绝原因：CRC 校验失败 vs 字段解包失败
            if verify_checksum and not _frame_crc32_ok(buf):
                crc_rejected_count += 1
        offset += FRAME_SIZE

    if not rows:
        empty = pd.DataFrame()
        empty.attrs.update({
            "binary_total_slots": n_frames_total,
            "binary_valid_frames": 0,
            "binary_bad_sync_frames": bad_sync_count,
            "binary_rejected_frames": rejected_count,
            "binary_crc_rejected_frames": crc_rejected_count,
            "binary_trailing_bytes": file_size % FRAME_SIZE,
            "binary_checksum_verified": bool(verify_checksum),
        })
        return empty

    df = pd.DataFrame(rows, columns=[
        "msg_id", "data_len", "timestamp_ms", "temp",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "pitch", "roll", "yaw"
    ])
    # 记录 id/length 字段取值分布（质量审计用，不做门限拒绝）
    df.attrs.update({
        "binary_total_slots": n_frames_total,
        "binary_valid_frames": valid_count,
        "binary_bad_sync_frames": bad_sync_count,
        "binary_rejected_frames": rejected_count,
        "binary_crc_rejected_frames": crc_rejected_count,
        "binary_trailing_bytes": file_size % FRAME_SIZE,
        "binary_checksum_verified": bool(verify_checksum),
        "binary_msg_id_values": sorted(df["msg_id"].astype(int).unique().tolist()),
        "binary_data_len_values": sorted(df["data_len"].astype(int).unique().tolist()),
    })
    return df


def calculate_frame_gap_statistics(frame_values, sample_rate: float,
                                   modulus: int = FRAME_COUNTER_MODULUS) -> dict:
    """统计帧计数器可推断出的缺口。

    这里统计的是 *计数器序号缺口*，不是对串口丢包原因的判定。计数器按
    1..60000 循环；相邻有效整数之间的正向步长 d>1 记为一处缺口，推定
    缺失 d-1 帧。超过半个计数器周期的跳变不强行解释为缺帧，而是单独列为
    不确定跳变。无效值也会切断连续判断。计数器本身无法识别整整一个或多个
    完整周期的缺失，也无法判断文件首尾的缺失，因此结果始终是基于
    可观测计数跳变的估计，不等同于实际传输丢包数。
    """
    result = {
        "available": False,
        "reason": "未检测到有效帧计数",
        "observed_frames": 0,
        "gap_events": 0,
        "missing_frames": 0,
        "expected_frames": 0,
        "missing_ratio": float("nan"),
        "received_ratio": float("nan"),
        "missing_duration_s": float("nan"),
        "observed_duration_s": float("nan"),
        "expected_duration_s": float("nan"),
        "counter_span_s": float("nan"),
        "duplicate_pairs": 0,
        "wrap_count": 0,
        "invalid_values": 0,
        "ambiguous_transitions": 0,
        "transition_count": 0,
        "ratio_is_partial_estimate": False,
        "modulus": int(modulus),
    }

    if frame_values is None:
        return result

    try:
        values = pd.to_numeric(frame_values, errors="coerce").to_numpy(dtype=float)
    except AttributeError:
        values = pd.to_numeric(pd.Series(frame_values), errors="coerce").to_numpy(dtype=float)

    valid = (np.isfinite(values) & (values == np.floor(values)) &
             (values >= FRAME_COUNTER_MIN) & (values <= modulus))
    result["invalid_values"] = int(np.count_nonzero(~valid))
    result["observed_frames"] = int(np.count_nonzero(valid))
    if result["observed_frames"] == 0:
        return result

    # 只比较原始序列中彼此相邻且都有效的记录；无效值不能被悄悄跨过。
    vals = np.zeros(values.shape, dtype=np.int64)
    vals[valid] = values[valid].astype(np.int64, copy=False)
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid):
        result["reason"] = "没有相邻的有效帧计数对，无法审计计数连续性"
        return result

    prev = vals[:-1][pair_valid]
    curr = vals[1:][pair_valid]
    raw_delta = curr - prev
    step = np.mod(raw_delta, int(modulus))

    # 大于半个周期的转移无法仅靠循环计数器区分“超长缺口”和复位/乱序。
    ambiguous = step > (int(modulus) // 2)
    trusted = ~ambiguous
    gap_mask = trusted & (step > 1)

    missing_frames = int(np.sum(step[gap_mask] - 1)) if np.any(gap_mask) else 0
    gap_events = int(np.count_nonzero(gap_mask))
    duplicate_pairs = int(np.count_nonzero(trusted & (step == 0)))
    wrap_count = int(np.count_nonzero(trusted & (raw_delta < 0)))
    ambiguous_count = int(np.count_nonzero(ambiguous))

    result.update({
        "available": True,
        "reason": "按帧计数器正向步长统计",
        "gap_events": gap_events,
        "missing_frames": missing_frames,
        "duplicate_pairs": duplicate_pairs,
        "wrap_count": wrap_count,
        "ambiguous_transitions": ambiguous_count,
        "transition_count": int(np.count_nonzero(pair_valid)),
        "ratio_is_partial_estimate": bool(result["invalid_values"] or ambiguous_count),
    })

    # 重复帧不增加“完整序列中的位置数”；未知跳变存在时，这个分母仅是已知部分。
    known_observed = result["observed_frames"] - duplicate_pairs
    expected_frames = known_observed + missing_frames
    result["expected_frames"] = int(max(expected_frames, 0))
    if sample_rate > 0 and result["expected_frames"] > 0:
        fs = float(sample_rate)
        result["missing_duration_s"] = missing_frames / fs
        result["observed_duration_s"] = known_observed / fs
        result["expected_duration_s"] = result["expected_frames"] / fs
        result["counter_span_s"] = max(result["expected_frames"] - 1, 0) / fs
        result["missing_ratio"] = missing_frames / result["expected_frames"]
        result["received_ratio"] = known_observed / result["expected_frames"]
    return result


def calculate_timestamp_gap_statistics(ts_values, sample_rate: float,
                                       modulus: int = TIMESTAMP_MODULUS) -> dict:
    """基于设备时间戳 (uint32 ms) 的缺帧审计。

    huace AHRS 帧内没有循环帧计数器（id 为恒定消息ID），但每帧携带
    uint32 毫秒时间戳。这里以相邻时间戳的中位步长 dt 为基准步长：
      - 步长 > dt 记为一处缺口，推定缺失 round(step/dt - 1) 帧；
      - 步长 == dt 视为连续；
      - 步长 == 0 记为重复帧（不增加缺失）；
      - 时间戳回绕按模 2^32 处理（约 49.7 天回绕一次）。
    推定缺帧数对 step/dt 取整存在 ±1 帧的不确定性；结果是基于可观测
    时间戳跳变的估计，不等同于实际传输丢包数。
    """
    result = {
        "available": False,
        "reason": "未检测到有效设备时间戳",
        "observed_frames": 0,
        "gap_events": 0,
        "missing_frames": 0,
        "expected_frames": 0,
        "missing_ratio": float("nan"),
        "received_ratio": float("nan"),
        "missing_duration_s": float("nan"),
        "observed_duration_s": float("nan"),
        "expected_duration_s": float("nan"),
        "counter_span_s": float("nan"),
        "duplicate_pairs": 0,
        "wrap_count": 0,
        "invalid_values": 0,
        "ambiguous_transitions": 0,
        "transition_count": 0,
        "ratio_is_partial_estimate": False,
        "modulus": int(modulus),
        "median_step_ms": float("nan"),
        "inferred_rate_hz": float("nan"),
        "min_step_ms": float("nan"),
        "max_step_ms": float("nan"),
        # 丢帧明细：每个缺口一条记录（最多返回 max_gap_details 条）：
        # {frame_index, ts_before_ms, ts_after_ms, step_ms, missing_frames, missing_duration_s}
        "gap_details": [],
        "gap_details_truncated": False,
    }

    if ts_values is None:
        return result

    try:
        values = pd.to_numeric(ts_values, errors="coerce").to_numpy(dtype=float)
    except AttributeError:
        values = pd.to_numeric(pd.Series(ts_values), errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(values) & (values >= 0) & (values < modulus)
    result["invalid_values"] = int(np.count_nonzero(~valid))
    result["observed_frames"] = int(np.count_nonzero(valid))
    if result["observed_frames"] < 2:
        return result

    vals = np.zeros(values.shape, dtype=np.int64)
    vals[valid] = values[valid].astype(np.int64, copy=False)
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid):
        result["reason"] = "没有相邻的有效时间戳对，无法审计连续性"
        return result

    prev = vals[:-1][pair_valid]
    curr = vals[1:][pair_valid]
    raw_delta = curr - prev
    step = np.mod(raw_delta, int(modulus))

    # 回绕：原始差为负且模后步长较小
    wrap_count = int(np.count_nonzero((raw_delta < 0) & (step < int(modulus) // 2)))

    # 以中位步长为基准 dt（对丢帧鲁棒）
    dt = float(np.median(step))
    if dt <= 0:
        result["reason"] = "时间戳中位步长非正，无法审计连续性"
        return result

    result["median_step_ms"] = dt
    result["inferred_rate_hz"] = 1000.0 / dt
    result["min_step_ms"] = float(np.min(step))
    result["max_step_ms"] = float(np.max(step))

    # 与基准步长的偏差超过半帧即视为异常步进
    gap_mask = step > dt * 1.5
    # 推定缺失帧数：按基准步长折算，向下取整
    missing_frames = int(np.sum(np.floor(step[gap_mask] / dt) - 1)) if np.any(gap_mask) else 0
    gap_events = int(np.count_nonzero(gap_mask))
    duplicate_pairs = int(np.count_nonzero(step == 0))

    # 丢帧位置明细（frame_index 为缺口前一帧在当前序列中的行号，
    # 即第 frame_index 帧与第 frame_index+1 帧之间出现缺口）
    max_gap_details = 500
    gap_details = []
    if np.any(gap_mask):
        pair_positions = np.nonzero(pair_valid)[0]  # 压缩索引 -> 原始帧行号
        gap_positions = pair_positions[gap_mask]
        gap_steps = step[gap_mask]
        fs = float(sample_rate) if sample_rate and sample_rate > 0 else 1.0
        for pos, gstep in zip(gap_positions[:max_gap_details], gap_steps[:max_gap_details]):
            g_missing = int(np.floor(gstep / dt) - 1)
            gap_details.append({
                "frame_index": int(pos),
                "ts_before_ms": int(vals[pos]),
                "ts_after_ms": int(vals[pos + 1]),
                "step_ms": int(gstep),
                "missing_frames": g_missing,
                "missing_duration_s": g_missing / fs,
            })
    result["gap_details"] = gap_details
    result["gap_details_truncated"] = bool(gap_events > max_gap_details)

    result.update({
        "available": True,
        "reason": "按设备时间戳步长统计（基准为中位步长）",
        "gap_events": gap_events,
        "missing_frames": missing_frames,
        "duplicate_pairs": duplicate_pairs,
        "wrap_count": wrap_count,
        "transition_count": int(np.count_nonzero(pair_valid)),
        "ratio_is_partial_estimate": bool(result["invalid_values"]),
    })

    known_observed = result["observed_frames"] - duplicate_pairs
    expected_frames = known_observed + missing_frames
    result["expected_frames"] = int(max(expected_frames, 0))
    if sample_rate > 0 and result["expected_frames"] > 0:
        fs = float(sample_rate)
        result["missing_duration_s"] = missing_frames / fs
        result["observed_duration_s"] = known_observed / fs
        result["expected_duration_s"] = result["expected_frames"] / fs
        result["counter_span_s"] = max(result["expected_frames"] - 1, 0) / fs
        result["missing_ratio"] = missing_frames / result["expected_frames"]
        result["received_ratio"] = known_observed / result["expected_frames"]
    # 时间戳自身给出的真实时长（与采样率无关）
    result["timestamp_span_s"] = float(step.sum()) / 1000.0
    return result


def detect_file_encoding(fp):
    """自动检测文件编码"""
    try:
        with open(fp, "rb") as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            if encoding in ["GB2312", "GBK", "ANSI"]:
                return "gbk"
            elif encoding == "UTF-8-SIG":
                return "utf-8-sig"
            else:
                return encoding
    except:
        return "gbk"


def detect_file_type(filepath: str) -> str:
    """
    检测文件类型：binary (huace AHRS .bin) 或 csv
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".bin":
        with open(filepath, "rb") as fh:
            header = fh.read(2)
        if header == FRAME_SYNC_BYTES:
            return "binary"
    return "csv"


class IMUDataAnalyzer:
    """IMU数据分析核心类 — 分析方法严格继承自 IMU_Analysis_yuanshen.py"""

    def __init__(self, file_path, sample_rate=1000, trim_minutes=0, allan_method="adev"):
        self.file_path = file_path
        self.sample_rate = float(sample_rate)
        self.trim_minutes = float(trim_minutes)  # 单位：分钟，0 表示不掐头去尾
        # 默认使用经典非重叠 ADEV，保持与现有厂家对照图的计算口径；
        # 可由程序调用方切换为 oadev，但报告会明确记录估计器名称。
        self.allan_method = str(allan_method).lower() if str(allan_method).lower() in {"adev", "oadev"} else "adev"
        self.save_dir = os.path.join(os.path.dirname(file_path), "analysis_results")
        self.report_data = {}
        self.frame_gap_stats = {}
        self.source_frame_gap_stats = {}
        self.binary_quality = {}

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._file_type = detect_file_type(file_path)
        self.load_data()

    # ----------------------------------------------------------------
    # 数据加载 (支持 CSV 和二进制)
    # ----------------------------------------------------------------
    def load_data(self):
        if self._file_type == "binary":
            self._load_binary_data()
        else:
            self._load_csv_data()
        self._apply_trim()  # 掐头去尾（必须在 extract_metadata 之前）
        self._update_frame_gap_statistics()
        self.extract_metadata()

    def _apply_trim(self):
        """掐头去尾：丢弃开头和结尾各 N 分钟的数据"""
        if self.trim_minutes <= 0 or self.sample_rate <= 0:
            return  # 不需要截断
        if self.df is None or len(self.df) == 0:
            return

        trim_samples = int(self.trim_minutes * 60 * self.sample_rate)
        total_samples = len(self.df)

        if trim_samples * 2 >= total_samples:
            print(f"  警告：掐头去尾 {self.trim_minutes} 分钟超过数据总长，不执行截断")
            return

        original_len = len(self.df)
        self.df = self.df.iloc[trim_samples:-trim_samples].reset_index(drop=True)
        discarded = original_len - len(self.df)
        print(f"  掐头去尾 {self.trim_minutes} 分钟：丢弃开头 {trim_samples} 点 + 结尾 {trim_samples} 点，保留 {len(self.df)} 点")
        self.report_data["掐头去尾"] = f"丢弃头尾各 {self.trim_minutes} 分钟 (共 {discarded} 点 / {original_len} 点，保留率 {len(self.df)/original_len*100:.1f}%)"
        self.report_data["掐头去尾口径"] = (
            "按已收到的观测行数裁剪；若时间戳存在缺口，设置的分钟数与实际设备时间可能不同，"
            "请同时参考按时间戳推定时长。"
        )

    def _update_frame_gap_statistics(self):
        """在实际分析数据上更新缺帧摘要。

        huace AHRS 优先使用设备时间戳 (timestamp_ms, uint32 ms) 审计；
        CSV 数据若含帧计数列则沿用计数器审计。
        """
        if "timestamp_ms" in self.df.columns:
            self.frame_gap_stats = calculate_timestamp_gap_statistics(
                self.df["timestamp_ms"], self.sample_rate, TIMESTAMP_MODULUS
            )
        else:
            frame_col = None
            for candidate in ("frameCount", "frame_id", "frame_count"):
                if candidate in self.df.columns:
                    frame_col = candidate
                    break

            if frame_col is None:
                self.frame_gap_stats = {
                    "available": False,
                    "reason": "未检测到设备时间戳或帧序号列",
                }
                self.report_data["缺帧统计"] = "未检测到时间戳/帧计数列，无法估计缺帧"
                return self.frame_gap_stats

            self.frame_gap_stats = calculate_frame_gap_statistics(
                self.df[frame_col], self.sample_rate, FRAME_COUNTER_MODULUS
            )
        stats = self.frame_gap_stats
        self.report_data["缺帧统计范围"] = "当前分析数据（已完成坏值清理及掐头去尾）"

        if self.binary_quality:
            total_slots = self.binary_quality.get("binary_total_slots", 0)
            rejected = self.binary_quality.get("binary_rejected_frames", 0)
            bad_sync = self.binary_quality.get("binary_bad_sync_frames", 0)
            trailing = self.binary_quality.get("binary_trailing_bytes", 0)
            crc_rejected = self.binary_quality.get("binary_crc_rejected_frames", 0)
            checksum_state = (
                "已启用（CRC32，算法经全量数据实测验证）"
                if self.binary_quality.get("binary_checksum_verified")
                else "未启用"
            )
            self.report_data["BIN帧质量"] = (
                f"固定槽位 {total_slots}，校验拒绝 {rejected}（其中 CRC32 拒绝 {crc_rejected}），"
                f"同步头异常 {bad_sync}，尾部余字节 {trailing}，校验验证{checksum_state}"
            )
            msg_ids = self.binary_quality.get("binary_msg_id_values")
            data_lens = self.binary_quality.get("binary_data_len_values")
            if msg_ids is not None:
                self.report_data["消息ID取值"] = ", ".join(str(v) for v in msg_ids)
            if data_lens is not None:
                self.report_data["length字段取值"] = ", ".join(str(v) for v in data_lens)

        if not stats.get("available", False):
            self.report_data["缺帧统计"] = stats.get("reason", "无法统计")
            return stats

        missing = int(stats.get("missing_frames", 0))
        events = int(stats.get("gap_events", 0))
        ratio = stats.get("missing_ratio", float("nan"))
        expected_s = stats.get("expected_duration_s", float("nan"))
        missing_s = stats.get("missing_duration_s", float("nan"))
        qualifier = (
            "（存在未计入的不确定跳变或非法值；仅为已知部分估计）"
            if stats.get("ratio_is_partial_estimate") else ""
        )
        if np.isfinite(ratio):
            ratio_text = f"{ratio * 100:.4f}%"
        else:
            ratio_text = "无法计算"

        self.report_data["缺帧处数"] = f"{events} 处"
        self.report_data["推定缺帧总数"] = f"{missing} 帧"
        self.report_data["缺帧占推定完整时长"] = (
            f"{ratio_text}{qualifier}（约 {missing_s:.3f} s / {expected_s:.3f} s）"
            if np.isfinite(missing_s) and np.isfinite(expected_s)
            else f"{ratio_text}{qualifier}"
        )
        if "timestamp_ms" in self.df.columns:
            median_step = stats.get("median_step_ms", float("nan"))
            inferred = stats.get("inferred_rate_hz", float("nan"))
            if np.isfinite(median_step):
                self.report_data["时间戳中位步长"] = f"{median_step:.3f} ms"
            if np.isfinite(inferred):
                self.report_data["时间戳推定采样率"] = f"{inferred:.2f} Hz"
            self.report_data["缺帧统计说明"] = (
                "按相邻设备时间戳步长统计：以中位步长为基准，步长超基准1.5倍记为一处缺口，"
                "推定缺失 floor(step/dt)-1 帧；时间戳回绕按模 2^32 处理。"
                "推定缺帧数存在 ±1 帧取整不确定性；这不是对串口/采集丢包原因"
                "或绝对丢包总数的判定。"
            )
        else:
            self.report_data["缺帧统计说明"] = (
                "按 frameCount 1～60000 循环、相邻正向步长 d>1 计为一处，"
                "推定缺失 d-1 帧；仅反映可观测计数跳变。整周期缺失、文件首尾缺失以及"
                "重复帧与整周期缺失的混淆无法仅由 frameCount 区分，因此这不是对串口/采集"
                "丢包原因或绝对丢包总数的判定。"
            )
        if stats.get("duplicate_pairs", 0):
            self.report_data["重复帧计数对"] = f"{stats['duplicate_pairs']} 对（不计入缺帧）"
        if stats.get("ambiguous_transitions", 0):
            self.report_data["不确定帧计数跳变"] = f"{stats['ambiguous_transitions']} 处（未计入缺帧总数）"
        if stats.get("invalid_values", 0):
            self.report_data["非法帧计数值"] = f"{stats['invalid_values']} 个（未跨越统计）"

        # 掐头去尾时同时保留源文件全量摘要，避免把分析窗口误解为原始文件总况。
        source = self.source_frame_gap_stats
        if source.get("available") and (
                source.get("observed_frames") != stats.get("observed_frames") or
                self.trim_minutes > 0):
            source_ratio = source.get("missing_ratio", float("nan"))
            source_ratio_text = (
                f"{source_ratio * 100:.4f}%" if np.isfinite(source_ratio) else "无法计算"
            )
            self.report_data["原始文件缺帧摘要"] = (
                f"{source.get('gap_events', 0)} 处，推定 {source.get('missing_frames', 0)} 帧，"
                f"占推定完整时长 {source_ratio_text}"
            )
        return stats

    def _load_binary_data(self):
        """加载 huace AHRS 二进制文件（启用 CRC32 校验）"""
        self.df = parse_binary_file(self.file_path, verify_checksum=True)
        if self.df.empty:
            raise ValueError("二进制文件解析结果为空，请检查文件是否完整"
                             "（huace AHRS 协议，54字节/帧，同步字 A5 D5，CRC32 已启用校验）")
        self.binary_quality = dict(self.df.attrs)
        self.source_frame_gap_stats = calculate_timestamp_gap_statistics(
            self.df["timestamp_ms"], self.sample_rate, TIMESTAMP_MODULUS
        )
        # 二进制数据已经标准化为 acc_x/acc_y/acc_z/gyro_x/gyro_y/gyro_z
        # 时间列使用设备时间戳（ms -> s，相对首帧），比按行号推定更贴近真实采样时刻
        ts = self.df["timestamp_ms"].to_numpy(dtype=np.float64)
        # 处理 uint32 回绕（当前数据未回绕；保险起见按累计单调展开）
        dt_arr = np.diff(ts)
        wraps = dt_arr < -(TIMESTAMP_MODULUS // 2)
        if np.any(wraps):
            ts[1:] += np.cumsum(np.where(wraps, TIMESTAMP_MODULUS, 0))
        self.df["time"] = (ts - ts[0]) / 1000.0

        # 数据完整性检测的源级数据（基于原始全量文件，不受掐头去尾影响）
        self._integrity_source = {
            "ts_unwrapped": ts.copy(),              # 回绕展开后的时间戳 (ms)
            "quality": dict(self.binary_quality),   # 帧质量（含 CRC 拒绝计数）
            "stats": dict(self.source_frame_gap_stats),  # 时间戳缺帧统计（含明细）
        }

    def _load_csv_data(self):
        """加载 CSV 数据 - 增强鲁棒性（参考原版）
        支持多种 CSV 格式：
        1. 常规 dump 格式：ax/ay/az/gx/gy/gz
        2. 标准格式：acc_x/acc_y/acc_z/gyro_x/gyro_y/gyro_z
        3. 中文格式：加速度X/加速度Y/加速度Z/角速度X/角速度Y/角速度Z
        """
        encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]

        self.df = None
        read_exception = None
        for enc in encodings:
            try:
                self.df = pd.read_csv(
                    self.file_path,
                    encoding=enc,
                    skipinitialspace=True,
                    low_memory=False,
                    on_bad_lines="error",
                )
                print(f"成功使用编码 {enc} 加载数据")
                break
            except Exception as e:
                read_exception = e
                print(f"使用编码 {enc} 失败: {e}")
                continue

        if self.df is None:
            raise ValueError(
                "无法完整加载CSV文件，请检查编码、列数和坏行。程序不会静默跳过坏行，"
                f"以免把解析失败误记为设备缺帧。最后错误: {read_exception}"
            )
        self.report_data["CSV坏行策略"] = "严格报错；不静默跳过，以免把解析失败混入缺帧统计"

        self.df = self.df.loc[:, ~self.df.columns.astype(str).str.match(r"^Unnamed")]
        self.df.columns = self.df.columns.astype(str).str.strip()
        self.df = self.df.loc[:, self.df.columns != ""]

        # 不同导出工具对帧计数列的命名不一致，统一为 frameCount。
        frame_counter_aliases = {
            "framecount", "frame_count", "frame-id", "frame_id",
            "帧序号", "帧计数", "帧号"
        }
        for column in list(self.df.columns):
            if str(column).strip().lower() in frame_counter_aliases:
                if column != "frameCount" and "frameCount" not in self.df.columns:
                    self.df = self.df.rename(columns={column: "frameCount"})
                break

        if "frameCount" in self.df.columns:
            # 在传感器列清洗前保留源文件层面的序号审计结果。
            self.source_frame_gap_stats = calculate_frame_gap_statistics(
                self.df["frameCount"], self.sample_rate, FRAME_COUNTER_MODULUS
            )

        # ── 第零步：列名标准化映射 ──
        # 常规 dump 格式：ax/ay/az → acc_x/acc_y/acc_z
        DUMP_CSV_COL_MAP = {
            "ax": "acc_x", "ay": "acc_y", "az": "acc_z",
            "gx": "gyro_x", "gy": "gyro_y", "gz": "gyro_z",
        }
        for old_name, new_name in DUMP_CSV_COL_MAP.items():
            if old_name in self.df.columns and new_name not in self.df.columns:
                self.df = self.df.rename(columns={old_name: new_name})
                print(f"  列名映射: {old_name} -> {new_name}")

        # 中文命名 -> 标准命名
        CN_COL_MAP = {
            "加速度X": "acc_x", "加速度Y": "acc_y", "加速度Z": "acc_z",
            "角速度X": "gyro_x", "角速度Y": "gyro_y", "角速度Z": "gyro_z",
            "加速度计X": "acc_x", "加速度计Y": "acc_y", "加速度计Z": "acc_z",
            "陀螺仪X": "gyro_x", "陀螺仪Y": "gyro_y", "陀螺仪Z": "gyro_z",
        }
        for old_name, new_name in CN_COL_MAP.items():
            if old_name in self.df.columns and new_name not in self.df.columns:
                self.df = self.df.rename(columns={old_name: new_name})
                print(f"  中文列名映射: {old_name} -> {new_name}")

        # 四元数列名标准化
        QUAT_COL_MAP = {
            "四元数q0": "Q0", "四元数q1": "Q1", "四元数q2": "Q2", "四元数q3": "Q3",
        }
        for old_name, new_name in QUAT_COL_MAP.items():
            if old_name in self.df.columns and new_name not in self.df.columns:
                self.df = self.df.rename(columns={old_name: new_name})

        # ── 第一步：检查必需列 ──
        required_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        available_cols = []
        missing_cols = []

        for col in required_cols:
            found = False
            if col in self.df.columns:
                available_cols.append(col)
                found = True
            else:
                variations = [col.lower(), col.upper(), col.replace("_", " ")]
                for v in variations:
                    if v in self.df.columns:
                        self.df = self.df.rename(columns={v: col})
                        available_cols.append(col)
                        found = True
                        break
            if not found:
                missing_cols.append(col)

        if len(available_cols) < 3:
            raise ValueError(f"缺少必要列！需要的列: {required_cols}\n找到的列: {list(self.df.columns)}")

        for col in available_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            except:
                pass

        self.df = self.df.dropna(subset=available_cols).reset_index(drop=True)

        if len(self.df) < 100:
            raise ValueError(f"有效数据点太少！仅 {len(self.df)} 个点")

    # ----------------------------------------------------------------
    # 数据基本信息
    # ----------------------------------------------------------------
    def extract_metadata(self):
        """提取数据基本信息"""
        self.n_points = len(self.df)
        self.duration = self.n_points / self.sample_rate

        self.report_data["采样率"] = f"{self.sample_rate} Hz"
        if "timestamp_ms" in self.df.columns and self.frame_gap_stats.get("available"):
            inferred = self.frame_gap_stats.get("inferred_rate_hz", float("nan"))
            if np.isfinite(inferred):
                self.report_data["采样率来源"] = (
                    f"用户输入/程序配置；设备时间戳推定采样率 {inferred:.2f} Hz，"
                    "可与用户配置交叉核对"
                )
            else:
                self.report_data["采样率来源"] = "用户输入/程序配置"
        else:
            self.report_data["采样率来源"] = (
                "用户输入/程序配置；帧序号只提供序号，不能单独确定采样频率"
            )
        self.report_data["总数据点数"] = f"{self.n_points} 点"
        self.report_data["采样时长"] = f"{self.duration:.2f} 秒 ({self.duration/60:.2f} 分钟)"
        if self.frame_gap_stats.get("available"):
            expected_s = self.frame_gap_stats.get("expected_duration_s", float("nan"))
            span_s = self.frame_gap_stats.get("counter_span_s", float("nan"))
            ts_span_s = self.frame_gap_stats.get("timestamp_span_s", float("nan"))
            if np.isfinite(expected_s):
                self.report_data["按时间戳/帧计数推定时长"] = f"{expected_s:.2f} 秒 ({expected_s/60:.2f} 分钟)"
            if np.isfinite(span_s):
                self.report_data["首末跨度"] = f"{span_s:.2f} 秒"
            if np.isfinite(ts_span_s):
                self.report_data["设备时间戳总跨度"] = f"{ts_span_s:.2f} 秒 ({ts_span_s/60:.2f} 分钟)"
        self.report_data["数据文件"] = os.path.basename(self.file_path)
        self.report_data["分析时间"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "time" in self.df.columns:
            try:
                self.report_data["时间范围"] = f"{self.df['time'].iloc[0]:.2f} ~ {self.df['time'].iloc[-1]:.2f} s"
            except:
                pass

        temp_cols = []
        # 优先精确匹配 temp 列名
        for col in self.df.columns:
            col_l = col.lower()
            if col_l == "temp" or "温度" in col:
                temp_cols.append(col)
        # 如果没有精确 temp，再尝试包含 temp 的其他列（如 temp_raw）
        if not temp_cols:
            for col in self.df.columns:
                if "temp" in col.lower():
                    temp_cols.append(col)

        if temp_cols:
            self.temp_col = temp_cols[0]
            try:
                temp_data = pd.to_numeric(self.df[self.temp_col], errors="coerce")
                temp_data = temp_data.dropna()
                if len(temp_data) > 0:
                    self.report_data["温度范围"] = f"{temp_data.min():.2f} ~ {temp_data.max():.2f}°C"
                    self.report_data["平均温度"] = f"{temp_data.mean():.2f}°C"
            except:
                self.report_data["温度数据"] = "读取失败"
        else:
            self.report_data["温度数据"] = "未检测到"

        for col in self.df.columns:
            if "timestamp" in col.lower() or "时间戳" in col:
                try:
                    t = pd.to_datetime(self.df[col], errors="coerce").dropna()
                    if len(t) > 0:
                        self.report_data["开始时间"] = t.min().strftime("%Y-%m-%d %H:%M:%S")
                        self.report_data["结束时间"] = t.max().strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass

    # ----------------------------------------------------------------
    # 基本统计
    # ----------------------------------------------------------------
    def calculate_basic_stats(self):
        """计算基本统计信息"""
        stats = {}

        G0 = 9.80665

        for axis in ["x", "y", "z"]:
            col = f"acc_{axis}"
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats[f"加速度计{axis.upper()}轴"] = {
                        "均值": f"{data.mean():.6f} m/s²",
                        "标准差": f"{data.std():.6f} m/s²",
                        "RMS": f"{np.sqrt(np.mean(data**2)):.6f} m/s²"
                    }

        for axis in ["x", "y", "z"]:
            col = f"gyro_{axis}"
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    stats[f"陀螺仪{axis.upper()}轴"] = {
                        "均值": f"{data.mean():.6f} °/s",
                        "标准差": f"{data.std():.6f} °/s",
                        "RMS": f"{np.sqrt(np.mean(data**2)):.6f} °/s"
                    }

        self.report_data["统计摘要"] = stats
        return stats

    # ----------------------------------------------------------------
    # 时间序列图
    # ----------------------------------------------------------------
    def plot_time_series(self):
        """绘制时间序列图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            if "acc_z" in self.df.columns:
                self.df["acc_z_corrected"] = self.df["acc_z"] - self.df["acc_z"].mean()  # 自适应去重力：减去Z轴实测均值（重力+零偏），兼容Z轴朝上/朝下安装

            if "time" not in self.df.columns:
                self.df["time"] = np.arange(len(self.df)) / self.sample_rate

            fig, axes = plt.subplots(4, 2, figsize=(16, 14))

            color_map = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}

            for i, axis in enumerate(["x", "y", "z"]):
                col = f"acc_{axis}"
                if col in self.df.columns:
                    data_col = col if axis != "z" else "acc_z_corrected"
                    if data_col in self.df.columns:
                        axes[i, 0].plot(self.df["time"], self.df[data_col],
                                       color=color_map[axis], linewidth=0.5, alpha=0.8)
                axes[i, 0].set_title(f"加速度计{axis.upper()}轴时间序列", fontweight="bold", fontsize=12)
                axes[i, 0].set_ylabel("加速度 (m/s²)", fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)

                col = f"gyro_{axis}"
                if col in self.df.columns:
                    axes[i, 1].plot(self.df["time"], self.df[col],
                                   color=color_map[axis], linewidth=0.5, alpha=0.8)
                axes[i, 1].set_title(f"陀螺仪{axis.upper()}轴时间序列", fontweight="bold", fontsize=12)
                axes[i, 1].set_ylabel("角速度 (°/s)", fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)

            for axis in ["x", "y", "z"]:
                col = f"acc_{axis}"
                if col in self.df.columns:
                    data_col = col if axis != "z" else "acc_z_corrected"
                    if data_col in self.df.columns:
                        axes[3, 0].plot(self.df["time"], self.df[data_col],
                                       label=f"{axis.upper()}", color=color_map[axis],
                                       linewidth=0.5, alpha=0.8)

            axes[3, 0].set_title("加速度计三轴时间序列（三合一）", fontweight="bold", fontsize=12)
            axes[3, 0].legend(loc="upper right", fontsize=9)
            axes[3, 0].set_ylabel("加速度 (m/s²)", fontsize=10)
            axes[3, 0].grid(True, alpha=0.3)

            for axis in ["x", "y", "z"]:
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    axes[3, 1].plot(self.df["time"], self.df[col],
                                   label=f"{axis.upper()}", color=color_map[axis],
                                   linewidth=0.5, alpha=0.8)

            axes[3, 1].set_title("陀螺仪三轴时间序列（三合一）", fontweight="bold", fontsize=12)
            axes[3, 1].legend(loc="upper right", fontsize=9)
            axes[3, 1].set_ylabel("角速度 (°/s)", fontsize=10)
            axes[3, 1].grid(True, alpha=0.3)

            for ax in axes[3, :]:
                ax.set_xlabel("时间 (s)", fontsize=10)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "01_时间序列图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["时间序列图"] = save_path
            return save_path
        except Exception as e:
            print(f"时间序列图绘制失败: {e}")
            self.report_data["时间序列图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # 统计分布图
    # ----------------------------------------------------------------
    def plot_distribution(self):
        """绘制统计分布图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            if "acc_z" in self.df.columns:
                self.df["acc_z_corrected"] = self.df["acc_z"] - self.df["acc_z"].mean()  # 自适应去重力：减去Z轴实测均值（重力+零偏），兼容Z轴朝上/朝下安装

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes_flat = axes.flatten()

            sub_titles = ["加速度计X轴静态分布", "加速度计Y轴静态分布", "加速度计Z轴(去重力)静态分布",
                         "陀螺仪X轴静态分布", "陀螺仪Y轴静态分布", "陀螺仪Z轴静态分布"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#1f77b4", "#ff7f0e", "#2ca02c"]

            for i, (axis, sensor) in enumerate([("x", "acc"), ("y", "acc"), ("z", "acc"),
                                               ("x", "gyro"), ("y", "gyro"), ("z", "gyro")]):
                col = f"{sensor}_{axis}"
                data_col = col
                if sensor == "acc" and axis == "z" and "acc_z_corrected" in self.df.columns:
                    data_col = "acc_z_corrected"

                if data_col in self.df.columns:
                    data = self.df[data_col].dropna()
                    if len(data) > 0:
                        sns.histplot(data=data, ax=axes_flat[i], kde=True, bins=50,
                                   color=colors[i], edgecolor="black", linewidth=0.5)
                        mean_val = float(data.mean())
                        std_val = float(data.std())
                        axes_flat[i].axvline(mean_val, color="red", linestyle="--", linewidth=1)
                        axes_flat[i].axvline(mean_val + 3 * std_val, color="orange",
                                            linestyle=":", linewidth=1, alpha=0.5)
                        axes_flat[i].axvline(mean_val - 3 * std_val, color="orange",
                                            linestyle=":", linewidth=1, alpha=0.5)
                        text_unit = "m/s²" if sensor == "acc" else "°/s"
                        axes_flat[i].text(0.02, 0.98,
                                         f"均值: {mean_val:.6f}\n标准差: {std_val:.6f}",
                                         transform=axes_flat[i].transAxes,
                                         verticalalignment="top",
                                         fontsize=8,
                                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

                axes_flat[i].set_title(sub_titles[i], fontweight="bold", fontsize=10)
                if sensor == "acc":
                    axes_flat[i].set_xlabel("加速度 (m/s²)", fontsize=8)
                else:
                    axes_flat[i].set_xlabel("角速度 (°/s)", fontsize=8)
                axes_flat[i].grid(True, alpha=0.2)
                axes_flat[i].set_ylabel("频次", fontsize=8)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "02_统计分布分析图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["统计分布图"] = save_path
            return save_path
        except Exception as e:
            print(f"分布图绘制失败: {e}")
            self.report_data["统计分布图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # Allan 偏差（ADEV/OADEV）参数提取
    # ----------------------------------------------------------------
    # 显示曲线使用较密的对数 tau 网格；参数提取仍固定使用 octave 网格。
    # 两套网格必须保持独立，避免仅为增加图形细节而改变 BI/RW 的既有口径。
    ALLAN_DISPLAY_POINTS_PER_DECADE = 15
    ALLAN_DISPLAY_GRID_NAME = "nominal_15_points_per_decade_plus_octave_anchors"
    ALLAN_ESTIMATION_GRID_NAME = "octave"
    ALLAN_DISPLAY_NORMAL_MIN_TERMS = 20
    ALLAN_DISPLAY_LOW_MIN_TERMS = 5

    @staticmethod
    def _build_allan_display_tau_grid(sample_count, sample_rate,
                                      points_per_decade=15):
        """生成显示用 tau：约每十倍程15点，并显式并入全部 octave 锚点。

        返回值以秒为单位。先在整数平均因子 ``m`` 上去重，保证传给
        AllanTools 后不会因为 tau 四舍五入产生重复计算点。
        """
        n = int(sample_count)
        rate = float(sample_rate)
        density = int(points_per_decade)
        if n < 2 or not np.isfinite(rate) or rate <= 0 or density <= 0:
            return np.array([], dtype=float)

        # 采用保守上限 floor(N/3)，保证经典非重叠 ADEV 的显示点
        # 至少有 2 个支持项；避免把最末端单支持项请求交给 AllanTools
        # 后再静默丢弃，也与元申版本保持一致。
        max_m = max(1, n // 3)
        max_log10_m = float(np.log10(max_m))
        log_count = max(1, int(np.ceil(max_log10_m * density)) + 1)
        log_m = np.rint(
            np.logspace(0.0, max_log10_m, num=log_count, base=10.0)
        ).astype(np.int64)
        max_octave_power = int(np.floor(np.log2(max_m)))
        octave_m = np.left_shift(
            np.int64(1), np.arange(max_octave_power + 1, dtype=np.int64)
        )
        averaging_factors = np.unique(np.concatenate((log_m, octave_m)))
        averaging_factors = averaging_factors[
            (averaging_factors >= 1) & (averaging_factors <= max_m)
        ]
        return averaging_factors.astype(float) / rate

    @classmethod
    def _build_log_tau_grid(cls, sample_count, sample_rate, points_per_decade=15):
        """兼容外部/回归测试的显示 tau 网格别名。"""
        return cls._build_allan_display_tau_grid(
            sample_count, sample_rate, points_per_decade
        )

    @classmethod
    def _allan_support_masks(cls, n_terms):
        """返回显示置信分组：正常、提醒、低支持（互斥且覆盖全部点）。"""
        ns = np.asarray(n_terms, dtype=float)
        normal = np.isfinite(ns) & (ns >= cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS)
        caution = (
            np.isfinite(ns) & (ns >= cls.ALLAN_DISPLAY_LOW_MIN_TERMS) &
            (ns < cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS)
        )
        # AllanTools 正常返回有限 n_terms；若上游异常给出 NaN，也保守归入低支持。
        low = ~(normal | caution)
        return normal, caution, low

    @classmethod
    def _allan_support_class(cls, n_terms):
        """给曲线 CSV 返回可机读的支持项分组。"""
        if np.isfinite(n_terms) and n_terms >= cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS:
            return "normal_n_terms_ge_20"
        if np.isfinite(n_terms) and n_terms >= cls.ALLAN_DISPLAY_LOW_MIN_TERMS:
            return "limited_n_terms_5_to_19"
        return "low_n_terms_lt_5"

    def extract_random_walk_coefficient(self, tau, ad, ns=None,
                                        return_details: bool = False):
        """在白噪声斜率区拟合随机游走系数。

        ADEV 的白噪声区斜率应接近 -1/2。返回的 ``value`` 是拟合曲线在
        tau=1 s 的系数；斜率明显不符时保留诊断信息但不把结果标成有效。
        ``return_details=False`` 保留旧调用方得到浮点数的兼容行为。
        """
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        mask = np.isfinite(tau) & np.isfinite(ad) & (tau > 0) & (ad > 0)
        if ns is not None:
            ns_arr = np.asarray(ns, dtype=float)
            if len(ns_arr) == len(mask):
                mask &= np.isfinite(ns_arr) & (ns_arr >= 3)

        mask &= (tau >= 0.1) & (tau <= 10.0)
        result = {
            "value": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "tau_min": float("nan"),
            "tau_max": float("nan"),
            "n_points": int(np.count_nonzero(mask)),
            "valid": False,
            "reason": "白噪声拟合区有效点不足（需要至少3点）",
        }
        if result["n_points"] >= 3:
            log_tau = np.log10(tau[mask])
            log_ad = np.log10(ad[mask])
            slope, intercept = np.polyfit(log_tau, log_ad, 1)
            result.update({
                "value": float(10 ** intercept),
                "slope": float(slope),
                "intercept": float(intercept),
                "tau_min": float(np.min(tau[mask])),
                "tau_max": float(np.max(tau[mask])),
            })
            if abs(slope + 0.5) <= 0.3:
                result["valid"] = True
                result["reason"] = "斜率满足本程序白噪声区工程判据（非标准强制门槛）"
            else:
                result["reason"] = f"拟合斜率 {slope:.3f} 偏离 -0.5，未作为有效 RW"

        if return_details:
            return result
        return result["value"] if result["valid"] else float("nan")

    # Flicker/pink rate-noise 平台关系：sigma_platform = 0.66428 * B。
    FLICKER_ADEV_FACTOR = float(np.sqrt(2.0 * np.log(2.0) / np.pi))
    BI_REFERENCE_TAU_MIN_S = 1.0
    BI_REFERENCE_MIN_TERMS = 20
    BI_PLATFORM_TAU_MIN_S = 1.0
    BI_PLATFORM_MIN_TERMS = 5
    BI_PLATFORM_POINTS = 3
    BI_PLATFORM_MAX_ABS_SLOPE = 0.10
    BI_PLATFORM_MAX_ABS_ADJACENT_SLOPE = 0.15
    BI_PLATFORM_MAX_LOG10_RMS = 0.03

    def extract_minimum_adev_reference(self, tau, ad, ns=None):
        """返回受约束的最低 ADEV 及其 BI 等效参考值。

        该值只是在 tau>=1 s 且 n_terms>=20 的候选点中取最低 ADEV，
        并按 ``B_ref = sigma_min / 0.66428`` 折算。它不会被标记为正式 BI，
        也不用于规格合格判定；正式 BI 仍由连续近零斜率平台提取。
        """
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        if ns is None:
            ns_arr = np.full(len(tau), np.nan, dtype=float)
        else:
            ns_arr = np.asarray(ns, dtype=float)
            if len(ns_arr) != len(tau):
                ns_arr = np.full(len(tau), np.nan, dtype=float)

        mask = (np.isfinite(tau) & np.isfinite(ad) & (tau >= self.BI_REFERENCE_TAU_MIN_S) &
                (ad > 0) & np.isfinite(ns_arr) &
                (ns_arr >= self.BI_REFERENCE_MIN_TERMS))
        result = {
            "available": False,
            "adev": float("nan"),
            "tau": float("nan"),
            "n_terms": float("nan"),
            "equivalent_bias": float("nan"),
            "local_slope": float("nan"),
            "edge_limited": False,
            "valid_for_spec_comparison": False,
            "tau_min_constraint_s": float(self.BI_REFERENCE_TAU_MIN_S),
            "min_terms_constraint": int(self.BI_REFERENCE_MIN_TERMS),
            "reason": (
                f"未找到同时满足 tau>={self.BI_REFERENCE_TAU_MIN_S:g} s 且 "
                f"n_terms>={self.BI_REFERENCE_MIN_TERMS} 的有限 ADEV 点"
            ),
        }
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return result

        selected = int(indices[np.argmin(ad[indices])])
        eligible_pos = int(np.where(indices == selected)[0][0])
        # 局部斜率只使用原始曲线上与最低点真正连续的候选点，避免跨过
        # NaN、低支持项或乱序点后仍把两侧数据拼成一个诊断窗口。
        local_indices = [selected]
        if eligible_pos > 0 and indices[eligible_pos - 1] == selected - 1:
            local_indices.insert(0, int(indices[eligible_pos - 1]))
        if (eligible_pos + 1 < len(indices) and
                indices[eligible_pos + 1] == selected + 1):
            local_indices.append(int(indices[eligible_pos + 1]))
        local_indices = np.asarray(local_indices, dtype=int)
        local_mask = (np.isfinite(tau[local_indices]) & np.isfinite(ad[local_indices]) &
                      (tau[local_indices] > 0) & (ad[local_indices] > 0))
        local_slope = float("nan")
        if (np.count_nonzero(local_mask) >= 2 and
                np.all(np.diff(tau[local_indices][local_mask]) > 0)):
            local_slope = float(np.polyfit(
                np.log10(tau[local_indices][local_mask]),
                np.log10(ad[local_indices][local_mask]), 1
            )[0])
        edge_limited = bool(eligible_pos == 0 or eligible_pos == len(indices) - 1)
        reason = (
            f"受约束最低ADEV（tau>={self.BI_REFERENCE_TAU_MIN_S:g} s，"
            f"n_terms>={self.BI_REFERENCE_MIN_TERMS}）；仅作等效参考，不是平台BI"
        )
        if edge_limited:
            reason += "；最低点位于候选范围边界"
        result.update({
            "available": True,
            "curve_index": selected,
            "adev": float(ad[selected]),
            "tau": float(tau[selected]),
            "n_terms": float(ns_arr[selected]),
            "equivalent_bias": float(ad[selected] / self.FLICKER_ADEV_FACTOR),
            "local_slope": local_slope,
            "edge_limited": edge_limited,
            "reason": reason,
        })
        return result

    def extract_bias_instability(self, tau, ad, ns=None):
        """识别连续近零斜率平台并提取 flicker BI。

        不再无条件取全局最低点。只有在 tau>=1 s、至少3个连续对数点、
        近似零斜率且支持项数足够时才给出 BI；否则返回 ``valid=False``。
        平台值与 BI 的关系为 ``B = sigma_platform / 0.66428``。
        """
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        if ns is None:
            ns_arr = np.full(len(tau), np.nan, dtype=float)
        else:
            ns_arr = np.asarray(ns, dtype=float)
            if len(ns_arr) != len(tau):
                ns_arr = np.full(len(tau), np.nan, dtype=float)

        mask = (np.isfinite(tau) & np.isfinite(ad) &
                (tau >= self.BI_PLATFORM_TAU_MIN_S) &
                (ad > 0) & np.isfinite(ns_arr) &
                (ns_arr >= self.BI_PLATFORM_MIN_TERMS))
        result = {
            "bias_instability": float("nan"),
            "min_adev": float("nan"),
            "tau_at_min": float("nan"),
            "platform_adev": float("nan"),
            "platform_tau": float("nan"),
            "fit_slope": float("nan"),
            "fit_residual": float("nan"),
            "fit_max_abs_adjacent_slope": float("nan"),
            "fit_tau_min": float("nan"),
            "fit_tau_max": float("nan"),
            "fit_points": 0,
            "fit_min_n_terms": float("nan"),
            "valid": False,
            "edge_limited": False,
            "low_edge_limited": False,
            "high_edge_limited": False,
            "platform_quality_passed": False,
            "valid_for_spec_comparison": False,
            "reason": (
                "没有足够的连续近零斜率平台；正式BI需要有效"
                "n_terms诊断"
            ),
        }
        indices = np.flatnonzero(mask)
        if len(indices) < self.BI_PLATFORM_POINTS:
            return result

        candidates = []
        window_size = self.BI_PLATFORM_POINTS
        for eligible_start in range(0, len(indices) - window_size + 1):
            window_indices = indices[eligible_start:eligible_start + window_size]
            # 不允许过滤后把跨 NaN/低支持点的 tau 拼成“连续平台”。
            if not np.all(np.diff(window_indices) == 1):
                continue
            t_win = tau[window_indices]
            a_win = ad[window_indices]
            n_win = ns_arr[window_indices]
            if not np.all(np.diff(t_win) > 0):
                continue
            log_t = np.log10(t_win)
            log_a = np.log10(a_win)
            adjacent_slopes = np.diff(log_a) / np.diff(log_t)
            slope, intercept = np.polyfit(log_t, log_a, 1)
            fitted = slope * log_t + intercept
            residual = float(np.sqrt(np.mean((log_a - fitted) ** 2)))
            max_adjacent = float(np.max(np.abs(adjacent_slopes)))
            if (abs(slope) <= self.BI_PLATFORM_MAX_ABS_SLOPE and
                    max_adjacent <= self.BI_PLATFORM_MAX_ABS_ADJACENT_SLOPE and
                    residual <= self.BI_PLATFORM_MAX_LOG10_RMS):
                # 显式排序：先拟合质量，再支持项数，最后较早 tau。
                # round 避免纯浮点噪声改变完全等价窗口的选择。
                score = round(abs(float(slope)) + residual, 12)
                candidates.append((score, -float(np.min(n_win)),
                                   int(window_indices[0]), window_indices,
                                   float(slope), float(intercept), residual,
                                   max_adjacent))
        if not candidates:
            result["reason"] = (
                "未找到同时满足的连续平台：|3点拟合斜率|<=0.10、"
                "max|相邻斜率|<=0.15、log10残差RMS<=0.03（本项目工程判据）"
            )
            return result

        _, _, _, selected_indices, slope, intercept, residual, max_adjacent = min(candidates)
        t_win = tau[selected_indices]
        n_win = ns_arr[selected_indices]
        center_log_tau = float(np.mean(np.log10(t_win)))
        platform_adev = float(10 ** (intercept + slope * center_log_tau))
        platform_tau = float(10 ** center_log_tau)
        bias = platform_adev / self.FLICKER_ADEV_FACTOR
        low_edge_limited = bool(selected_indices[0] == indices[0])
        high_edge_limited = bool(selected_indices[-1] == indices[-1])
        edge_limited = low_edge_limited or high_edge_limited
        reason = (
            f"连续平台拟合（log斜率={slope:.3f}，"
            f"max|相邻斜率|={max_adjacent:.3f}，残差={residual:.3g}）；"
            "平台门槛是本项目工程判据，非标准强制值"
        )
        if low_edge_limited:
            first_index = int(selected_indices[0])
            has_in_search_points_before = bool(np.any(
                np.isfinite(tau[:first_index]) &
                (tau[:first_index] >= self.BI_PLATFORM_TAU_MIN_S)
            ))
            if has_in_search_points_before:
                reason += (
                    "；平台位于有效候选范围左边界；此前有 tau 位于搜索域，"
                    "但 ADEV 或 n_terms 未满足有效性门槛"
                )
            else:
                reason += (
                    "；平台位于有效候选范围左边界并与 tau 搜索下限相接，"
                    "需扩大搜索域或复核短 tau 噪声区"
                )
        if high_edge_limited:
            reason += "；平台受记录长度/支持项上界限制，需更长数据确认"
        result.update({
            "bias_instability": float(bias),
            "min_adev": platform_adev,  # 旧字段名兼容；实际是平台拟合值
            "tau_at_min": platform_tau,
            "platform_adev": platform_adev,
            "platform_tau": platform_tau,
            "fit_slope": slope,
            "fit_residual": residual,
            "fit_max_abs_adjacent_slope": max_adjacent,
            "fit_tau_min": float(t_win[0]),
            "fit_tau_max": float(t_win[-1]),
            "fit_points": int(window_size),
            "fit_min_n_terms": float(np.min(n_win)),
            "valid": True,
            "edge_limited": edge_limited,
            "low_edge_limited": low_edge_limited,
            "high_edge_limited": high_edge_limited,
            "platform_quality_passed": True,
            # 边界平台虽可以报告拟合值，但不应在没有更长数据确认时
            # 用作规格合格判定。
            # 即便不是边界平台，也需另行匹配厂家测试条件与定义。
            "valid_for_spec_comparison": False,
            "reason": reason,
        })
        return result

    # ----------------------------------------------------------------
    # 10 s 非重叠分段均值标准差（工程统计量）
    # ----------------------------------------------------------------
    def calculate_bias_stability_10s(self):
        """按10 s非重叠窗口计算分段均值的总体标准差（ddof=0）。

        这是可追溯的工程统计量；未在程序中宣称对某一标准条款的合规性。
        """
        G0 = 9.80665
        fs = self.sample_rate
        window = max(1, int(round(10 * fs)))  # 10s 窗口样本数

        acc_bs_gjb = {}
        gyro_bs_gjb = {}
        acc_bs_raw = {}
        gyro_bs_raw = {}

        for axis in ["x", "y", "z"]:
            # 加速度计
            col = f"acc_{axis}"
            if col in self.df.columns:
                data = self.df[col].to_numpy(dtype=float, copy=False)
                data = data[np.isfinite(data)]
                if len(data) >= 2 * window:
                    n_seg = len(data) // window
                    seg_means = np.array([np.mean(data[i*window:(i+1)*window]) for i in range(n_seg)])
                    std_10s = np.std(seg_means, ddof=0)
                    acc_bs_raw[axis] = float(std_10s)
                    bias_stab_mg = std_10s * 1000 / G0
                    acc_bs_gjb[axis] = f"{bias_stab_mg:.6f} mg"

            # 陀螺仪
            col = f"gyro_{axis}"
            if col in self.df.columns:
                data = self.df[col].to_numpy(dtype=float, copy=False)
                data = data[np.isfinite(data)]
                if len(data) >= 2 * window:
                    n_seg = len(data) // window
                    seg_means = np.array([np.mean(data[i*window:(i+1)*window]) for i in range(n_seg)])
                    std_10s = np.std(seg_means, ddof=0)
                    gyro_bs_raw[axis] = float(std_10s)
                    bias_stab_dph = std_10s * 3600.0
                    gyro_bs_gjb[axis] = f"{bias_stab_dph:.6f} °/h"

        self.report_data["加速度计零偏稳定性_10s平滑"] = acc_bs_gjb
        self.report_data["陀螺仪零偏稳定性_10s平滑"] = gyro_bs_gjb
        self.report_data["加速度计BS_10s单位"] = {
            axis: {
                "m/s²": value,
                "g": value / G0,
                "mg": value * 1e3 / G0,
                "μg": value * 1e6 / G0,
            } for axis, value in acc_bs_raw.items()
        }
        self.report_data["陀螺仪BS_10s单位"] = {
            axis: {
                "°/s": value,
                "°/h": value * 3600.0,
                "rad/s": value * np.pi / 180.0,
                "rad/h": value * 20.0 * np.pi,
            } for axis, value in gyro_bs_raw.items()
        }
        return acc_bs_gjb, gyro_bs_gjb

    def calculate_bias_stability_gjb(self):
        """旧接口兼容包装；实际执行10 s分段均值标准差。"""
        return self.calculate_bias_stability_10s()

    def _gjb_10s_smoothing_std(self, data, window):
        """10 s 非重叠分段均值标准差。
        将数据按 window 个样本分段，计算各段均值的标准差。
        返回原始单位下的数值 (加速度计: m/s², 陀螺仪: °/s)，数据不足返回 NaN。
        """
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if window <= 0 or len(data) <= window:
            return float('nan')
        n_seg = len(data) // window
        if n_seg < 2:
            return float('nan')
        seg_means = np.array([np.mean(data[k * window:(k + 1) * window]) for k in range(n_seg)])
        return float(np.std(seg_means))

    # ----------------------------------------------------------------
    # Allan 偏差图
    # ----------------------------------------------------------------
    def plot_allan_variance(self):
        """绘制 Allan 偏差（ADEV/OADEV）并输出可追溯的参数单位。"""
        try:
            # 同一分析器对象重跑时，先移除上一次运行的路径和结果，
            # 避免本次失败后报告仍指向旧 CSV/旧参数。
            for stale_key in (
                "Allan曲线数据", "Allan参数汇总", "Allan方差图", "Allan偏差图",
                "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
            ):
                self.report_data.pop(stale_key, None)
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            G0 = 9.80665
            fs_float = float(self.sample_rate)
            method_name = str(getattr(self, "allan_method", "adev")).lower()
            if method_name not in {"adev", "oadev"}:
                method_name = "adev"
            estimator = at.oadev if method_name == "oadev" else at.adev
            method_label = "OADEV" if method_name == "oadev" else "ADEV（经典非重叠）"

            self.report_data["Allan分析配置"] = (
                f"估计器={method_label}；输入=全量观测行；采样率={fs_float:g} Hz；"
                "未线性去趋势、未插值；"
                "显示网格=每十倍程约15点并保留全部octave锚点；"
                "BI/RW/最低ADEV等效参考的参数提取网格=octave（既有判据不变）；"
                "n_terms≥20、5≤n_terms<20、n_terms<5的20/5分级仅为本项目"
                "工程可视化门槛，非标准规定。"
            )
            self.report_data["Allan曲线支持项显示"] = (
                "n_terms≥20正常实线；5≤n_terms<20淡色虚线；"
                "n_terms<5灰色低支持尾部；20/5为本项目工程可视化门槛，非标准规定"
            )
            self.report_data["Allan显示网格"] = (
                "nominal 15 points/decade plus octave anchors；仅增加显示和曲线CSV细节，"
                "未进行曲线平滑"
            )
            self.report_data["Allan参数提取网格"] = (
                "octave；BI、RW及最低ADEV等效参考均只使用该独立网格"
            )
            self.report_data["Allan预处理"] = (
                "原始速率序列（加速度 m/s²、陀螺 °/s）；未做线性去趋势。"
            )
            if not self.frame_gap_stats.get("available"):
                self.report_data["Allan缺帧策略"] = (
                    "未检测到有效设备时间戳/帧序号，无法判断缺帧；Allan 输入仍按观测行处理。"
                )
            elif self.frame_gap_stats.get("gap_events", 0):
                self.report_data["Allan缺帧策略"] = (
                    "未重建缺失时间点；按观测行兼容计算，并在报告中单独列出时间戳/帧序号缺口。"
                    "由于 AllanTools 假定等间隔采样，缺帧条件下曲线仅作兼容性参考，"
                    "不应替代基于可靠时间戳或连续段处理的计量级结果。"
                )
            else:
                uncertain = int(self.frame_gap_stats.get("ambiguous_transitions", 0))
                uncertain += int(self.frame_gap_stats.get("invalid_values", 0))
                if uncertain:
                    self.report_data["Allan缺帧策略"] = (
                        f"未发现可确认的正向缺口，但存在 {uncertain} 个非法值或不确定跳变；"
                        "缺帧结论仅覆盖可观测部分。"
                    )
                else:
                    self.report_data["Allan缺帧策略"] = "未发现可由时间戳/帧序号推断的正向缺口。"

            # 每个传感器轴使用独立的“曲线 + 数值说明”区域。曲线区固定
            # 为宽:高=6:4；说明文字不叠加在数据区域内。
            fig = plt.figure(figsize=(24, 21), constrained_layout=False)
            outer_grid = fig.add_gridspec(
                2, 3, left=0.045, right=0.985, top=0.975, bottom=0.115,
                wspace=0.18, hspace=0.22,
            )
            axes = []
            info_axes = []
            for plot_index in range(6):
                cell = outer_grid[plot_index // 3, plot_index % 3].subgridspec(
                    2, 1, height_ratios=[3.15, 1.85], hspace=0.20,
                )
                curve_ax = fig.add_subplot(cell[0])
                curve_ax.set_label(f"allan_curve_{plot_index}")
                curve_ax.set_box_aspect(4 / 6)
                info_ax = fig.add_subplot(cell[1])
                info_ax.set_label(f"allan_info_{plot_index}")
                info_ax.set_axis_off()
                axes.append(curve_ax)
                info_axes.append(info_ax)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            data_names = ['acc_x(m/s²)', 'acc_y(m/s²)', 'acc_z(m/s²)',
                          'gyro_x(°/s)', 'gyro_y(°/s)', 'gyro_z(°/s)']
            units = ["m/s²", "m/s²", "m/s²", "°/s", "°/s", "°/s"]
            curve_rows = []
            summary_rows = []
            acc_bias = {}
            gyro_bias = {}
            acc_results = {}
            gyro_results = {}
            frame_audit_available = bool(self.frame_gap_stats.get("available", False))
            frame_gap_events = (
                int(self.frame_gap_stats.get("gap_events", 0))
                if frame_audit_available else np.nan
            )
            frame_missing_frames = (
                int(self.frame_gap_stats.get("missing_frames", 0))
                if frame_audit_available else np.nan
            )

            window_10s = max(1, int(round(10 * fs_float)))
            acc_bs = {}
            gyro_bs = {}
            for axis in ["x", "y", "z"]:
                col = f"acc_{axis}"
                if col in self.df.columns:
                    acc_bs[axis] = self._gjb_10s_smoothing_std(
                        self.df[col].to_numpy(dtype=float, copy=False), window_10s)
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    gyro_bs[axis] = self._gjb_10s_smoothing_std(
                        self.df[col].to_numpy(dtype=float, copy=False), window_10s)

            def _fmt(value, unit="", digits=6):
                if value is None or not np.isfinite(value):
                    return "N/A"
                return f"{value:.{digits}e} {unit}".rstrip()

            def _acc_units(value):
                return {
                    "m/s²": value,
                    "g": value / G0 if np.isfinite(value) else np.nan,
                    "mg": value * 1e3 / G0 if np.isfinite(value) else np.nan,
                    "μg": value * 1e6 / G0 if np.isfinite(value) else np.nan,
                }

            def _gyro_units(value):
                return {
                    "°/s": value,
                    "°/h": value * 3600.0 if np.isfinite(value) else np.nan,
                    "rad/s": value * np.pi / 180.0 if np.isfinite(value) else np.nan,
                    "rad/h": value * 20.0 * np.pi if np.isfinite(value) else np.nan,
                }

            def _acc_rw_units(value):
                one_sided_asd = value * np.sqrt(2.0) if np.isfinite(value) else np.nan
                return {
                    "(m/s²)·√s": value,
                    "m/s/√s": value,
                    "m/s/√h": value * 60.0 if np.isfinite(value) else np.nan,
                    "mg·√s": value * 1e3 / G0 if np.isfinite(value) else np.nan,
                    "μg·√s": value * 1e6 / G0 if np.isfinite(value) else np.nan,
                    "单边ASD_(m/s²)/√Hz": one_sided_asd,
                    "单边ASD_mg/√Hz": (
                        one_sided_asd * 1e3 / G0 if np.isfinite(one_sided_asd) else np.nan
                    ),
                    "单边ASD_μg/√Hz": (
                        one_sided_asd * 1e6 / G0 if np.isfinite(one_sided_asd) else np.nan
                    ),
                }

            def _gyro_rw_units(value):
                one_sided_asd = value * np.sqrt(2.0) if np.isfinite(value) else np.nan
                return {
                    "°/√s": value,
                    "°/√h": value * 60.0 if np.isfinite(value) else np.nan,
                    "rad/√s": value * np.pi / 180.0 if np.isfinite(value) else np.nan,
                    "rad/√h": value * np.pi / 3.0 if np.isfinite(value) else np.nan,
                    "单边ASD_(°/s)/√Hz": one_sided_asd,
                }

            for i in range(6):
                ax = axes[i]
                info_ax = info_axes[i]
                sensor = "acc" if i < 3 else "gyro"
                axis = ["x", "y", "z"][i if i < 3 else i - 3]
                col = f"{sensor}_{axis}"
                ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                if col not in self.df.columns:
                    ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                    info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                 fontsize=7, color='#666666', transform=info_ax.transAxes)
                    continue

                try:
                    raw = self.df[col].to_numpy(dtype=np.float64, copy=False)
                    finite = np.isfinite(raw)
                    if np.count_nonzero(finite) < 100:
                        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue
                    if not np.all(finite):
                        self.report_data["Allan非有限样本"] = (
                            f"{int(np.count_nonzero(~finite))} 个；按有限观测行计算，未插值"
                        )
                    data = raw[finite]

                    display_tau_request = self._build_allan_display_tau_grid(
                        len(data), fs_float, self.ALLAN_DISPLAY_POINTS_PER_DECADE
                    )
                    taus, adev, adev_error, n_terms = estimator(
                        data, rate=fs_float, data_type="freq", taus=display_tau_request
                    )
                    parameter_taus, parameter_adev, parameter_error, parameter_n_terms = estimator(
                        data, rate=fs_float, data_type="freq",
                        taus=self.ALLAN_ESTIMATION_GRID_NAME
                    )
                    taus = np.asarray(taus, dtype=float)
                    adev = np.asarray(adev, dtype=float)
                    adev_error = np.asarray(adev_error, dtype=float)
                    n_terms = np.asarray(n_terms, dtype=float)
                    parameter_taus = np.asarray(parameter_taus, dtype=float)
                    parameter_adev = np.asarray(parameter_adev, dtype=float)
                    parameter_error = np.asarray(parameter_error, dtype=float)
                    parameter_n_terms = np.asarray(parameter_n_terms, dtype=float)
                    valid_curve = np.isfinite(taus) & np.isfinite(adev) & (taus > 0) & (adev > 0)
                    valid_parameter_curve = (
                        np.isfinite(parameter_taus) & np.isfinite(parameter_adev) &
                        (parameter_taus > 0) & (parameter_adev > 0)
                    )
                    if not np.any(valid_curve) or not np.any(valid_parameter_curve):
                        ax.text(0.5, 0.5, '计算失败', ha='center', va='center', transform=ax.transAxes)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue

                    taus_v = taus[valid_curve]
                    adev_v = adev[valid_curve]
                    err_v = adev_error[valid_curve] if len(adev_error) == len(taus) else np.full(len(taus_v), np.nan)
                    ns_v = n_terms[valid_curve] if len(n_terms) == len(taus) else np.full(len(taus_v), np.nan)
                    parameter_taus_v = parameter_taus[valid_parameter_curve]
                    parameter_adev_v = parameter_adev[valid_parameter_curve]
                    # 参数提取必须保留 octave Allan 数组的索引间断；否则先删除
                    # NaN/低支持项再提取，会把断点两侧误拼成连续平台。
                    ns_for_extract = (
                        parameter_n_terms
                        if len(parameter_n_terms) == len(parameter_taus)
                        else np.full(len(parameter_taus), np.nan)
                    )
                    rw_result = self.extract_random_walk_coefficient(
                        parameter_taus, parameter_adev,
                        ns=ns_for_extract, return_details=True
                    )
                    bi_result = self.extract_bias_instability(
                        parameter_taus, parameter_adev, ns=ns_for_extract
                    )
                    min_ref_result = self.extract_minimum_adev_reference(
                        parameter_taus, parameter_adev, ns=ns_for_extract
                    )
                    rw = rw_result["value"] if rw_result["valid"] else np.nan
                    bi = bi_result["bias_instability"] if bi_result["valid"] else np.nan
                    min_ref = (
                        min_ref_result["equivalent_bias"]
                        if min_ref_result["available"] else np.nan
                    )

                    normal_mask, caution_mask, low_mask = self._allan_support_masks(ns_v)
                    ax.loglog(
                        taus_v, np.where(normal_mask, adev_v, np.nan),
                        linewidth=1.2, linestyle='-', marker=None, color=colors[i],
                        label=f'{method_label}：n_terms≥20'
                    )
                    if np.any(caution_mask):
                        ax.loglog(
                            taus_v, np.where(caution_mask, adev_v, np.nan),
                            linewidth=1.1, linestyle='--', marker=None,
                            color=colors[i], alpha=0.48,
                            label='提醒：5≤n_terms<20'
                        )
                    if np.any(low_mask):
                        ax.loglog(
                            taus_v, np.where(low_mask, adev_v, np.nan),
                            linewidth=1.0, linestyle=':', marker=None,
                            color='#808080', alpha=0.9,
                            label='低支持尾部：n_terms<5'
                        )

                    idx_1s = int(np.argmin(np.abs(taus_v - 1.0)))
                    if taus_v.min() <= 1.0 <= taus_v.max():
                        ax.scatter(taus_v[idx_1s], adev_v[idx_1s], color='red', s=80,
                                   zorder=5, marker='o', edgecolors='darkred', linewidths=2,
                                   label=f'近1 s（实际 {taus_v[idx_1s]:.3g} s）')
                    if bi_result["valid"]:
                        ax.scatter(bi_result["platform_tau"], bi_result["platform_adev"],
                                   color='blue', s=80, zorder=5, marker='s',
                                   edgecolors='darkblue', linewidths=2,
                                   label='BI 平台拟合')
                    if min_ref_result["available"]:
                        ax.scatter(min_ref_result["tau"], min_ref_result["adev"],
                                   color='#e83e8c', s=90, zorder=6, marker='v',
                                   edgecolors='#8b1a50', linewidths=1.5,
                                   label='最低 ADEV 折算参考（非 BI）')

                    if sensor == "acc":
                        bs_value = acc_bs.get(axis, np.nan)
                        bi_units = _acc_units(bi)
                        min_ref_units = _acc_units(min_ref)
                        rw_units = _acc_rw_units(rw)
                        bs_units = _acc_units(bs_value)
                        bi_status = (
                            "边界受限（规格可比性未确认）"
                            if bi_result.get("edge_limited") else
                            ("平台识别通过（规格可比性未确认）"
                             if bi_result["valid"] else "不可提取")
                        )
                        info_text = (
                            f"VRW（斜率={rw_result['slope']:.3f}，"
                            f"{'有效' if rw_result['valid'] else '无效'}）:\n"
                            f"  {_fmt(rw_units['m/s/√s'], 'm/s/√s')}\n"
                            f"  {_fmt(rw_units['m/s/√h'], 'm/s/√h')}\n"
                            f"  {_fmt(rw_units['μg·√s'], 'μg·√s')}\n"
                            f"  单边白噪声ASD等效: "
                            f"{_fmt(rw_units['单边ASD_μg/√Hz'], 'μg/√Hz')}\n"
                            f"BI（平台 σ/0.66428，{bi_status}）:\n"
                            f"  {_fmt(bi_units['m/s²'], 'm/s²')}\n"
                            f"  {_fmt(bi_units['mg'], 'mg')} / {_fmt(bi_units['μg'], 'μg')}\n"
                            f"最低ADEV等效参考（非正式BI）:\n"
                            f"  {_fmt(min_ref_units['mg'], 'mg')} / {_fmt(min_ref_units['μg'], 'μg')}\n"
                            f"  tau={_fmt(min_ref_result['tau'], 's', 3)}, "
                            f"n={_fmt(min_ref_result['n_terms'], '', 0)}, "
                            f"局部斜率={_fmt(min_ref_result['local_slope'], '', 3)}"
                            f"{'（候选边界）' if min_ref_result['edge_limited'] else ''}\n"
                            f"BS（10 s分段均值标准差）:\n"
                            f"  {_fmt(bs_units['m/s²'], 'm/s²')}\n"
                            f"  {_fmt(bs_units['mg'], 'mg')} / {_fmt(bs_units['μg'], 'μg')}"
                        )
                        if bi_result["valid"] and bi_result.get("edge_limited"):
                            info_text = f"{bi_result['reason']}\n" + info_text
                        acc_bias[axis] = (_fmt(bi_units['mg'], 'mg')
                                          if bi_result['valid'] else "N/A（" + bi_result['reason'] + "）")
                        if bi_result["valid"] and bi_result.get("edge_limited"):
                            acc_bias[axis] += "（边界受限）"
                        acc_results[axis] = {
                            "rw": rw_result, "bi": bi_result, "min_ref": min_ref_result,
                            "rw_units": rw_units, "bi_units": bi_units,
                            "min_ref_units": min_ref_units,
                            "bs_units": bs_units,
                        }
                    else:
                        bs_value = gyro_bs.get(axis, np.nan)
                        bi_units = _gyro_units(bi)
                        min_ref_units = _gyro_units(min_ref)
                        rw_units = _gyro_rw_units(rw)
                        bs_units = _gyro_units(bs_value)
                        bi_status = (
                            "边界受限（规格可比性未确认）"
                            if bi_result.get("edge_limited") else
                            ("平台识别通过（规格可比性未确认）"
                             if bi_result["valid"] else "不可提取")
                        )
                        info_text = (
                            f"ARW（斜率={rw_result['slope']:.3f}，"
                            f"{'有效' if rw_result['valid'] else '无效'}）:\n"
                            f"  {_fmt(rw_units['°/√s'], '°/√s')}\n"
                            f"  {_fmt(rw_units['°/√h'], '°/√h')}\n"
                            f"  {_fmt(rw_units['rad/√s'], 'rad/√s')}\n"
                            f"  单边白噪声ASD等效: "
                            f"{_fmt(rw_units['单边ASD_(°/s)/√Hz'], '(°/s)/√Hz')}\n"
                            f"BI（平台 σ/0.66428，{bi_status}）:\n"
                            f"  {_fmt(bi_units['°/s'], '°/s')}\n"
                            f"  {_fmt(bi_units['°/h'], '°/h')} / {_fmt(bi_units['rad/h'], 'rad/h')}\n"
                            f"最低ADEV等效参考（非正式BI）:\n"
                            f"  {_fmt(min_ref_units['°/h'], '°/h')} / {_fmt(min_ref_units['rad/h'], 'rad/h')}\n"
                            f"  tau={_fmt(min_ref_result['tau'], 's', 3)}, "
                            f"n={_fmt(min_ref_result['n_terms'], '', 0)}, "
                            f"局部斜率={_fmt(min_ref_result['local_slope'], '', 3)}"
                            f"{'（候选边界）' if min_ref_result['edge_limited'] else ''}\n"
                            f"BS（10 s分段均值标准差）:\n"
                            f"  {_fmt(bs_units['°/s'], '°/s')}\n"
                            f"  {_fmt(bs_units['°/h'], '°/h')} / {_fmt(bs_units['rad/h'], 'rad/h')}"
                        )
                        if bi_result["valid"] and bi_result.get("edge_limited"):
                            info_text = f"{bi_result['reason']}\n" + info_text
                        gyro_bias[axis] = (_fmt(bi_units['°/h'], '°/h')
                                           if bi_result['valid'] else "N/A（" + bi_result['reason'] + "）")
                        if bi_result["valid"] and bi_result.get("edge_limited"):
                            gyro_bias[axis] += "（边界受限）"
                        gyro_results[axis] = {
                            "rw": rw_result, "bi": bi_result, "min_ref": min_ref_result,
                            "rw_units": rw_units, "bi_units": bi_units,
                            "min_ref_units": min_ref_units,
                            "bs_units": bs_units,
                        }

                    wrapped_info_lines = []
                    for info_line in info_text.splitlines():
                        if len(info_line) > 43:
                            wrapped_info_lines.extend(textwrap.wrap(
                                info_line, width=43, break_long_words=True,
                                break_on_hyphens=False,
                            ))
                        else:
                            wrapped_info_lines.append(info_line)
                    info_ax.text(
                        0.015, 0.985, "\n".join(wrapped_info_lines),
                        transform=info_ax.transAxes, fontsize=6.5,
                        linespacing=1.12, ha='left', va='top', clip_on=True,
                        wrap=True,
                        color='#2f2f2f',
                    )
                    ax.set_xlabel('tau (s)', fontsize=8)
                    # 保持原图纵轴单位不变：ADEV 与输入速率同量纲。
                    ax.set_ylabel(f'Allan Deviation ({units[i]})', fontsize=8)
                    ax.tick_params(axis='both', which='both', labelsize=7)
                    ax.grid(True, which='both', alpha=0.3)
                    ax.legend(fontsize=6.4, loc='best', framealpha=0.82,
                              borderpad=0.35, handlelength=2.4, labelspacing=0.3)

                    parameter_m = set(np.rint(parameter_taus_v * fs_float).astype(np.int64))
                    for j in range(len(taus_v)):
                        display_m = int(round(float(taus_v[j]) * fs_float))
                        curve_rows.append({
                            "sensor": sensor, "axis": axis,
                            "tau_s": float(taus_v[j]), "adev": float(adev_v[j]),
                            "error": float(err_v[j]) if np.isfinite(err_v[j]) else np.nan,
                            "n_terms": float(ns_v[j]) if np.isfinite(ns_v[j]) else np.nan,
                            "grid_role": "display_curve",
                            "tau_grid_role": (
                                "display_and_parameter_octave_anchor"
                                if display_m in parameter_m else "display_only"
                            ),
                            "display_tau_grid": self.ALLAN_DISPLAY_GRID_NAME,
                            "display_points_per_decade": self.ALLAN_DISPLAY_POINTS_PER_DECADE,
                            "estimation_tau_grid": self.ALLAN_ESTIMATION_GRID_NAME,
                            "parameter_estimation_source": "separate_octave_grid",
                            "is_parameter_tau": display_m in parameter_m,
                            "support_class": self._allan_support_class(ns_v[j]),
                            "display_support_class": self._allan_support_class(ns_v[j]),
                            "method": method_label, "sample_rate_hz": fs_float,
                            "sample_rate_source": "user_config_crosschecked_by_device_timestamp_ms",
                            "preprocessing": "raw_no_linear_detrend",
                            "gap_policy": "observed_rows_no_reconstruction",
                            "allan_error_semantics": "AllanTools_approximate_error_not_confidence_interval",
                            "frame_audit_available": frame_audit_available,
                            "frame_audit_reason": self.frame_gap_stats.get("reason", ""),
                            "frame_audit_scope": "current_analysis_rows_after_numeric_cleaning_and_trim",
                            "frame_gap_events": frame_gap_events,
                            "frame_missing_frames": frame_missing_frames,
                            "frame_missing_ratio": float(self.frame_gap_stats.get("missing_ratio", np.nan)),
                        })

                    summary_rows.append({
                        "sensor": sensor,
                        "axis": axis,
                        "input_unit": units[i],
                        "method": method_label,
                        "sample_rate_hz": fs_float,
                        "sample_rate_source": "user_config_crosschecked_by_device_timestamp_ms",
                        "preprocessing": "raw_no_linear_detrend",
                        "gap_policy": "observed_rows_no_reconstruction",
                        "display_tau_grid": self.ALLAN_DISPLAY_GRID_NAME,
                        "display_points_per_decade": self.ALLAN_DISPLAY_POINTS_PER_DECADE,
                        "estimation_tau_grid": self.ALLAN_ESTIMATION_GRID_NAME,
                        "parameter_estimation_source": "separate_octave_grid",
                        "allan_error_semantics": "AllanTools_approximate_error_not_confidence_interval",
                        "frame_audit_available": frame_audit_available,
                        "frame_audit_reason": self.frame_gap_stats.get("reason", ""),
                        "frame_audit_scope": "current_analysis_rows_after_numeric_cleaning_and_trim",
                        "tau_min_s": float(np.min(taus_v)),
                        "tau_max_s": float(np.max(taus_v)),
                        "curve_points": int(len(taus_v)),
                        "display_tau_min_s": float(np.min(taus_v)),
                        "display_tau_max_s": float(np.max(taus_v)),
                        "display_curve_points": int(len(taus_v)),
                        "parameter_tau_min_s": float(np.min(parameter_taus_v)),
                        "parameter_tau_max_s": float(np.max(parameter_taus_v)),
                        "parameter_curve_points": int(len(parameter_taus_v)),
                        "rw_valid": bool(rw_result["valid"]),
                        "rw_value_input_rate_times_sqrt_s": float(rw) if np.isfinite(rw) else np.nan,
                        "rw_one_sided_white_asd_equivalent_input_rate_per_sqrt_hz": (
                            float(np.sqrt(2.0) * rw) if np.isfinite(rw) else np.nan
                        ),
                        "rw_one_sided_white_asd_equivalent_unit": (
                            "(m/s^2)/sqrt(Hz)" if sensor == "acc" else "(deg/s)/sqrt(Hz)"
                        ),
                        "rw_one_sided_white_asd_equivalent_valid": bool(rw_result["valid"]),
                        "rw_one_sided_white_asd_equivalence_factor": float(np.sqrt(2.0)),
                        "rw_one_sided_white_asd_assumption": (
                            "white_rate_noise_and_one_sided_psd_from_0_to_fs_over_2"
                        ),
                        "rw_one_sided_white_asd_provenance": (
                            "derived_from_adev_rw_not_direct_psd_estimate"
                        ),
                        "rw_vendor_noise_density_comparable": False,
                        "rw_vendor_noise_density_comparability_reason": (
                            "vendor_psd_sidedness_and_definition_not_confirmed"
                        ),
                        "rw_slope": float(rw_result.get("slope", np.nan)),
                        "rw_reason": rw_result.get("reason", ""),
                        "bi_available": bool(bi_result["valid"]),
                        "bi_value_input_unit": float(bi) if np.isfinite(bi) else np.nan,
                        "bi_platform_adev": float(bi_result.get("platform_adev", np.nan)),
                        "bi_platform_tau_s": float(bi_result.get("platform_tau", np.nan)),
                        "bi_fit_slope": float(bi_result.get("fit_slope", np.nan)),
                        "bi_fit_residual_log10_rms": float(
                            bi_result.get("fit_residual", np.nan)
                        ),
                        "bi_fit_max_abs_adjacent_slope": float(
                            bi_result.get("fit_max_abs_adjacent_slope", np.nan)
                        ),
                        "bi_fit_tau_min_s": float(bi_result.get("fit_tau_min", np.nan)),
                        "bi_fit_tau_max_s": float(bi_result.get("fit_tau_max", np.nan)),
                        "bi_fit_points": int(bi_result.get("fit_points", 0)),
                        "bi_fit_min_n_terms": float(bi_result.get("fit_min_n_terms", np.nan)),
                        "bi_edge_limited": bool(bi_result.get("edge_limited", False)),
                        "bi_low_edge_limited": bool(bi_result.get("low_edge_limited", False)),
                        "bi_high_edge_limited": bool(bi_result.get("high_edge_limited", False)),
                        "bi_platform_quality_passed": bool(
                            bi_result.get("platform_quality_passed", False)
                        ),
                        "bi_valid_for_spec_comparison": bool(
                            bi_result.get("valid_for_spec_comparison", False)
                        ),
                        "bi_reason": bi_result.get("reason", ""),
                        "min_ref_available": bool(min_ref_result["available"]),
                        "min_ref_adev": float(min_ref_result.get("adev", np.nan)),
                        "min_ref_equivalent_bias": float(min_ref),
                        "min_ref_tau_s": float(min_ref_result.get("tau", np.nan)),
                        "min_ref_n_terms": float(min_ref_result.get("n_terms", np.nan)),
                        "min_ref_local_slope": float(min_ref_result.get("local_slope", np.nan)),
                        "min_ref_edge_limited": bool(min_ref_result.get("edge_limited", False)),
                        "min_ref_valid_for_spec_comparison": False,
                        "min_ref_tau_constraint_s": float(self.BI_REFERENCE_TAU_MIN_S),
                        "min_ref_min_terms_constraint": int(self.BI_REFERENCE_MIN_TERMS),
                        "min_ref_reason": min_ref_result.get("reason", ""),
                        "bs_10s_value_input_unit": float(bs_value) if np.isfinite(bs_value) else np.nan,
                        "frame_gap_events": frame_gap_events,
                        "frame_missing_frames": frame_missing_frames,
                        "frame_missing_ratio": float(self.frame_gap_stats.get("missing_ratio", np.nan)),
                    })
                except Exception as e:
                    print(f"{data_names[i]} Allan偏差计算失败: {e}")
                    ax.text(0.5, 0.5, f'计算失败\n{str(e)}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
                    info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                 fontsize=7, color='#666666', transform=info_ax.transAxes)

            conv_text = (
                "═══════════════════════════════════════════════════════════════════════════════════════════\n"
                "             Allan 偏差（ADEV）结果与常用单位换算参考\n"
                "───────────────────────────────────────────────────────────────────────────────────\n"
                "图中坐标保持：加速度 ADEV = m/s²，陀螺 ADEV = °/s，tau = s\n"
                "加速度 VRW系数N: m/s/√s → m/s/√h = ×60；→ μg·√s = ×1e6/9.80665\n"
                "陀螺 ARW系数N:   °/√s → °/√h = ×60\n"
                "           → rad/√s = ×π/180；rad/√h = ×π/3\n"
                "条件等效单边白噪声ASD = √2×N；仅限白速率噪声及0..fs/2单边PSD约定，\n"
                "不是直接谱估计，也不自动等同厂家噪声密度。\n"
                "加速度 BI/BS: m/s² → g = /9.80665；mg = ×1e3/9.80665；μg = ×1e6/9.80665\n"
                "陀螺 BI/BS:   °/s → °/h = ×3600；rad/s = ×π/180；rad/h = ×20π\n"
                "BI 仅在连续近零斜率平台上计算：B = σ平台 / 0.66428；无平台显示 N/A\n"
                "平台工程判据：3个原曲线连续点，|拟合斜率|<=0.10，"
                "max|相邻斜率|<=0.15，log10残差RMS<=0.03（非标准强制门槛）\n"
                "最低ADEV等效参考：仅在 tau>=1 s 且 n_terms>=20 中取最低点；"
                "非正式 BI，不用于规格判定\n"
                "曲线显示为名义15点/十倍程并加入octave锚点，不做后处理平滑；\n"
                "n_terms>=20正常实线，5..19淡色虚线，<5灰色低支持尾部。"
                "20/5是本项目工程可视化门槛，非标准规定。\n"
                "注：单位换算是量纲转换；缺帧未插值，缺帧统计见报告基本信息。\n"
                "═══════════════════════════════════════════════════════════════════════════════════════════"
            )
            fig.text(0.5, 0.012, conv_text, ha='center', va='bottom', fontsize=6.0,
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5dc',
                               edgecolor='#888888', alpha=0.95))
            save_path = os.path.join(self.save_dir, "03_零偏稳定性分析图.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
            plt.close()

            curve_path = os.path.join(self.save_dir, "03_Allan曲线数据.csv")
            if curve_rows:
                pd.DataFrame(curve_rows).to_csv(curve_path, index=False, encoding="utf-8-sig")
            else:
                curve_path = "无有效Allan曲线数据"

            summary_path = os.path.join(self.save_dir, "03_Allan参数汇总.csv")
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    summary_path, index=False, encoding="utf-8-sig"
                )
            else:
                summary_path = "无有效Allan参数汇总"

            self.report_data["加速度计零偏不稳定性_BI"] = acc_results
            self.report_data["陀螺仪零偏不稳定性_BI"] = gyro_results
            # 保留旧键，避免已有报告模板/外部调用失效；其内容仍是BI而非10 s BS。
            self.report_data["加速度计零偏稳定性"] = acc_bias
            self.report_data["陀螺仪零偏稳定性"] = gyro_bias
            self.report_data["Allan曲线数据"] = curve_path
            self.report_data["Allan参数汇总"] = summary_path
            self.report_data["Allan方差图"] = save_path  # 兼容旧键和文件名
            self.report_data["Allan偏差图"] = save_path
            return save_path
        except Exception as e:
            print(f"Allan偏差图绘制失败: {e}")
            import traceback as _tb
            _tb.print_exc()
            self.report_data["Allan方差图"] = "绘制失败"
            self.report_data["Allan偏差图"] = "绘制失败"
            self.report_data["Allan曲线数据"] = "生成失败"
            self.report_data["Allan参数汇总"] = "生成失败"
            self.report_data["加速度计零偏稳定性"] = {}
            self.report_data["陀螺仪零偏稳定性"] = {}
            self.report_data["加速度计零偏不稳定性_BI"] = {}
            self.report_data["陀螺仪零偏不稳定性_BI"] = {}
            return None

    # ----------------------------------------------------------------
    # 功率谱密度 (PSD)
    # ----------------------------------------------------------------
    def plot_psd(self):
        """绘制功率谱密度分析图（严格参考 IMU_Analysis_yuanshen.py）"""
        try:
            # 清空残留图形上下文（避免前一步骤的异常图形污染）
            plt.close("all")

            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            G0 = 9.80665

            def _calc_psd(data, fs, nperseg=1024):
                """内部 PSD 计算函数（参考原版）"""
                data = np.asarray(data)
                data = data[np.isfinite(data)]
                if len(data) < 10:
                    return None, None
                nperseg = int(min(nperseg, len(data)))
                noverlap = int(nperseg // 2)
                fs_float = float(fs) if fs is not None else self.sample_rate
                f, psd = welch(data, fs=fs_float, nperseg=nperseg, noverlap=noverlap,
                              window="hann", detrend="constant", scaling="density",
                              return_onesided=True)
                return f, psd

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            color_map = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
            acc_asd_factor = 1e6 / G0
            gyro_asd_factor = 1.0

            for sensor_type, row_idx in [("acc", 0), ("gyro", 1)]:
                for axis in ["x", "y", "z"]:
                    col = f"{sensor_type}_{axis}"
                    if col not in self.df.columns:
                        continue

                    data = self.df[col].dropna()
                    if len(data) <= 100:
                        continue

                    f, psd = _calc_psd(data, self.sample_rate)
                    if f is None:
                        continue

                    asd_factor = acc_asd_factor if sensor_type == "acc" else gyro_asd_factor
                    asd = np.sqrt(psd) * asd_factor

                    axes[row_idx, 0].semilogy(f, asd, color=color_map[axis],
                                              label=f"{axis.upper()}轴", linewidth=1)
                    axes[row_idx, 1].loglog(f[f > 0], asd[f > 0],
                                            color=color_map[axis],
                                            label=f"{axis.upper()}轴", linewidth=1)

            y_labels = {"acc": "ASD (µg/√Hz)", "gyro": "ASD (°/s/√Hz)"}
            titles = {"acc": "加速度计", "gyro": "陀螺仪"}

            for sensor_type, row_idx in [("acc", 0), ("gyro", 1)]:
                axes[row_idx, 0].set_title(f"{titles[sensor_type]}ASD（Y轴对数）",
                                           fontweight="bold", fontsize=13)
                axes[row_idx, 0].set_ylabel(y_labels[sensor_type], fontsize=11)
                axes[row_idx, 0].legend(fontsize=9)
                axes[row_idx, 0].grid(True, which="both", alpha=0.3)

                axes[row_idx, 1].set_title(f"{titles[sensor_type]}ASD（双对数）",
                                           fontweight="bold", fontsize=13)
                axes[row_idx, 1].legend(fontsize=9)
                axes[row_idx, 1].grid(True, which="both", alpha=0.3)

                axes[row_idx, 0].set_xlabel("频率 (Hz)", fontsize=11)
                axes[row_idx, 1].set_xlabel("频率 (Hz)", fontsize=11)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "04_PSD功率谱密度图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["PSD图"] = save_path
            return save_path
        except Exception as e:
            print(f"PSD图绘制失败: {e}")
            import traceback as _tb
            _tb.print_exc()
            self.report_data["PSD图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # 相关性分析
    # ----------------------------------------------------------------
    def plot_correlation(self):
        """绘制相关性热图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # 加速度计相关性
            acc_cols = [f"acc_{a}" for a in ["x", "y", "z"] if f"acc_{a}" in self.df.columns]
            if len(acc_cols) >= 2:
                acc_data = self.df[acc_cols].dropna().values
                if len(acc_data) > 10:
                    # 相关性展示使用原始观测值；不对数据施加线性去趋势。
                    acc_corr = np.corrcoef(acc_data, rowvar=False)
                    acc_labels = [c.replace("acc_", "A") for c in acc_cols]
                    sns.heatmap(acc_corr, annot=True, fmt=".4f", cmap="coolwarm",
                               center=0, vmin=-1, vmax=1, ax=axes[0],
                               xticklabels=acc_labels, yticklabels=acc_labels,
                               linewidths=0.5)
                axes[0].set_title("加速度计相关性热图", fontweight="bold", fontsize=14)
            else:
                axes[0].text(0.5, 0.5, "加速度计数据不足", ha="center", va="center",
                           transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title("加速度计相关性热图", fontweight="bold", fontsize=14)

            # 陀螺仪相关性
            gyro_cols = [f"gyro_{a}" for a in ["x", "y", "z"] if f"gyro_{a}" in self.df.columns]
            if len(gyro_cols) >= 2:
                gyro_data = self.df[gyro_cols].dropna().values
                if len(gyro_data) > 10:
                    # 相关性展示使用原始观测值；不对数据施加线性去趋势。
                    gyro_corr = np.corrcoef(gyro_data, rowvar=False)
                    gyro_labels = [c.replace("gyro_", "G") for c in gyro_cols]
                    sns.heatmap(gyro_corr, annot=True, fmt=".4f", cmap="YlGnBu",
                               center=0.5, vmin=0, vmax=1, ax=axes[1],
                               xticklabels=gyro_labels, yticklabels=gyro_labels,
                               linewidths=0.5)
                axes[1].set_title("陀螺仪相关性热图", fontweight="bold", fontsize=14)
            else:
                axes[1].text(0.5, 0.5, "陀螺仪数据不足", ha="center", va="center",
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title("陀螺仪相关性热图", fontweight="bold", fontsize=14)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "05_相关性分析图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["相关性图"] = save_path
            return save_path
        except Exception as e:
            print(f"相关性图绘制失败: {e}")
            self.report_data["相关性图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # 长期零漂趋势图
    # ----------------------------------------------------------------
    def plot_drift(self):
        """绘制长期零漂趋势图 (窗口式滑窗 + 四元数旋转 + 散点 + 1σ圆)"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            quat_candidates = [["四元数q0", "四元数q1", "四元数q2", "四元数q3"],
                              ["q0", "q1", "q2", "q3"],
                              ["Q0", "Q1", "Q2", "Q3"]]

            quat_cols = None
            for candidate in quat_candidates:
                if all(col in self.df.columns for col in candidate):
                    quat_cols = candidate
                    break

            if quat_cols is None:
                self.report_data["漂移分析图"] = "未检测到四元数数据"
                return None

            acc_cols = ["acc_x", "acc_y", "acc_z"]
            if not all(col in self.df.columns for col in acc_cols):
                self.report_data["漂移分析图"] = "缺少加速度计数据"
                return None

            WINDOW_SIZES = [5, 10, 30, 60, 120]  # 秒
            fs = self.sample_rate
            fs_int = int(round(fs))
            if fs <= 0 or fs_int <= 0:
                self.report_data["漂移分析图"] = "采样率无效"
                return None

            dt = 1.0 / fs
            step_size = fs_int

            acc_raw = self.df[acc_cols].values
            quats = self.df[quat_cols].values

            results = {sec: [] for sec in WINDOW_SIZES}
            for sec in WINDOW_SIZES:
                window_samples = sec * fs_int
                if len(self.df) < window_samples + 100:
                    continue

                for start in range(0, len(self.df) - window_samples, step_size):
                    end = start + window_samples
                    try:
                        r = R.from_quat(quats[start:end])
                        acc_w = r.apply(acc_raw[start:end])
                        acc_xy = acc_w[:, :2]
                        acc_xy_centered = acc_xy - np.mean(acc_xy, axis=0)

                        vel_x = np.cumsum(acc_xy_centered[:, 0]) * dt
                        vel_y = np.cumsum(acc_xy_centered[:, 1]) * dt

                        pos_x = np.cumsum(vel_x) * dt
                        pos_y = np.cumsum(vel_y) * dt

                        results[sec].append([pos_x[-1], pos_y[-1]])
                    except:
                        continue

            self._drift_results_cache = {
                "window_sizes": tuple(WINDOW_SIZES),
                "results": results
            }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for i, sec in enumerate(WINDOW_SIZES):
                endpoints = results[sec]
                if len(endpoints) == 0:
                    continue
                endpoints = np.array(endpoints)

                ax1.scatter(endpoints[:, 0], endpoints[:, 1], s=15, alpha=0.5,
                           color=colors[i], label=f"{sec}s")

                dist = np.linalg.norm(endpoints, axis=1)
                if len(dist) > 0:
                    sigma_1 = np.percentile(dist, 68)

                    circle = plt.Circle((0, 0), sigma_1, color=colors[i], fill=False,
                                       linestyle="--", linewidth=2,
                                       label=f"{sec}s 1σ: {sigma_1:.4f}m")
                    ax1.add_artist(circle)

                    ax2.plot(dist, color=colors[i], alpha=0.7, label=f"{sec}s")

            ax1.set_title("水平漂移散点图与1σ圆", fontweight="bold", fontsize=14)
            ax1.set_xlabel("X方向位移 (m)", fontsize=12)
            ax1.set_ylabel("Y方向位移 (m)", fontsize=12)
            ax1.grid(True)
            ax1.legend(loc="upper right")

            ax2.set_title("漂移量趋势图", fontweight="bold", fontsize=14)
            ax2.set_xlabel("滑动窗口索引", fontsize=12)
            ax2.set_ylabel("误差 (m)", fontsize=12)
            ax2.grid(True)
            ax2.legend(loc="upper right")

            all_data = []
            for sec in WINDOW_SIZES:
                data = np.array(results[sec])
                if len(data) > 0:
                    all_data.extend(data)

            if len(all_data) > 0:
                all_data = np.array(all_data)
                max_range = np.percentile(np.linalg.norm(all_data, axis=1), 95)
                max_range = max(max_range, 0.1)
                ax1.set_xlim(-max_range, max_range)
                ax1.set_ylim(-max_range, max_range)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "06_长期零漂趋势图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["漂移分析图"] = save_path
            return save_path
        except Exception as e:
            print(f"漂移分析图绘制失败: {e}")
            self.report_data["漂移分析图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # 不同时间窗口1σ落点半径对比
    # ----------------------------------------------------------------
    def plot_drift_radius_comparison(self):
        """绘制不同秒级时间窗口的1σ落点半径对比图。"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            window_seconds = [5, 10, 30, 60, 120]
            cached = getattr(self, "_drift_results_cache", None)
            if cached and all(sec in cached.get("results", {}) for sec in window_seconds):
                results = cached["results"]
            else:
                quat_candidates = [["四元数q0", "四元数q1", "四元数q2", "四元数q3"],
                                  ["q0", "q1", "q2", "q3"],
                                  ["Q0", "Q1", "Q2", "Q3"]]

                quat_cols = None
                for candidate in quat_candidates:
                    if all(col in self.df.columns for col in candidate):
                        quat_cols = candidate
                        break

                if quat_cols is None:
                    self.report_data["落点半径对比图"] = "未检测到四元数数据"
                    return None

                acc_cols = ["acc_x", "acc_y", "acc_z"]
                if not all(col in self.df.columns for col in acc_cols):
                    self.report_data["落点半径对比图"] = "缺少加速度计数据"
                    return None

                fs = self.sample_rate
                fs_int = int(round(fs))
                if fs <= 0 or fs_int <= 0:
                    self.report_data["落点半径对比图"] = "采样率无效"
                    return None

                dt = 1.0 / fs
                step_size = fs_int
                acc_raw = self.df[acc_cols].values
                quats = self.df[quat_cols].values
                results = {sec: [] for sec in window_seconds}

                for sec in window_seconds:
                    window_samples = sec * fs_int
                    if len(self.df) < window_samples + 100:
                        continue

                    for start in range(0, len(self.df) - window_samples, step_size):
                        end = start + window_samples
                        try:
                            r = R.from_quat(quats[start:end])
                            acc_w = r.apply(acc_raw[start:end])
                            acc_xy = acc_w[:, :2]
                            acc_xy_centered = acc_xy - np.mean(acc_xy, axis=0)

                            vel_x = np.cumsum(acc_xy_centered[:, 0]) * dt
                            vel_y = np.cumsum(acc_xy_centered[:, 1]) * dt

                            pos_x = np.cumsum(vel_x) * dt
                            pos_y = np.cumsum(vel_y) * dt

                            results[sec].append([pos_x[-1], pos_y[-1]])
                        except:
                            continue

                self._drift_results_cache = {
                    "window_sizes": tuple(window_seconds),
                    "results": results
                }

            radius_rows = []
            for sec in window_seconds:
                endpoints = results.get(sec, [])
                if len(endpoints) == 0:
                    continue

                endpoints = np.array(endpoints)
                distances = np.linalg.norm(endpoints, axis=1)
                radius_68 = np.percentile(distances, 68)
                radius_rows.append({
                    "second": sec,
                    "radius": radius_68,
                    "count": len(distances)
                })

            if not radius_rows:
                self.report_data["落点半径对比图"] = "数据量不足，无法计算5/10/30/60/120秒窗口"
                return None

            seconds = np.array([row["second"] for row in radius_rows], dtype=float)
            radii = np.array([row["radius"] for row in radius_rows], dtype=float)
            labels = [f"{int(s)}秒" for s in seconds]
            colors = ["#6BAED6", "#C17BA3", "#F4B24D", "#D9785F", "#8E75B8"][:len(radius_rows)]

            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.bar(seconds, radii, width=4.0, color=colors, edgecolor="#444444",
                         linewidth=1.2)

            max_radius = max(float(np.max(radii)), 0.001)
            for idx, (bar, radius) in enumerate(zip(bars, radii)):
                if radius >= 1:
                    value_text = f"{radius:.2f}m"
                elif radius >= 0.01:
                    value_text = f"{radius:.3f}m"
                else:
                    value_text = f"{radius:.4f}m"
                x_pos = bar.get_x() + bar.get_width() / 2
                xytext = (0, 6)
                ha = "center"
                if len(seconds) > 1 and seconds[1] - seconds[0] <= 10:
                    if idx == 0:
                        xytext = (-8, 6)
                        ha = "right"
                    elif idx == 1:
                        xytext = (8, 6)
                        ha = "left"
                ax.annotate(value_text, xy=(x_pos, bar.get_height()), xytext=xytext,
                            textcoords="offset points", ha=ha, va="bottom",
                            fontsize=11, color="#333333")

            if len(seconds) >= 2:
                fit_degree = 2 if len(seconds) >= 3 else 1
                coeffs = np.polyfit(seconds, radii, fit_degree)
                fit_x = np.linspace(seconds.min(), seconds.max(), 200)
                fit_y = np.polyval(coeffs, fit_x)
                ax.plot(fit_x, fit_y, color="red", linestyle="--", linewidth=2)

                if fit_degree == 2:
                    equation = f"趋势线: y={coeffs[0]:.3g}x²{coeffs[1]:+.3g}x{coeffs[2]:+.3g}"
                else:
                    equation = f"趋势线: y={coeffs[0]:.3g}x{coeffs[1]:+.3g}"
                ax.plot([], [], color="red", linestyle="--", linewidth=2, label=equation)
                ax.legend(loc="upper left", frameon=True)

            ax.set_title("不同时间窗口的1σ落点半径对比", fontweight="bold", fontsize=16)
            ax.set_xlabel("时间窗口 (秒)", fontweight="bold", fontsize=12)
            ax.set_ylabel("1σ落点半径 (m)", fontsize=12)
            ax.set_xticks(seconds)
            ax.set_xticklabels(labels)
            ax.grid(True, axis="y", alpha=0.3)
            ax.grid(True, axis="x", alpha=0.2)
            ax.set_ylim(0, max_radius * 1.18)

            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "07_不同时间窗口1σ落点半径对比图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.report_data["落点半径对比图"] = save_path
            self.report_data["落点半径对比数据"] = [
                {
                    "时间窗口": f"{row['second']}秒",
                    "1σ落点半径": f"{row['radius']:.6f} m",
                    "有效窗口数": row["count"]
                }
                for row in radius_rows
            ]
            return save_path
        except Exception as e:
            print(f"落点半径对比图绘制失败: {e}")
            self.report_data["落点半径对比图"] = "绘制失败"
            return None

    # ----------------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------------
    def _allan_report_cells(self, sensor: str, axis: str) -> dict:
        """把 Allan 结构化结果格式化为 HTML/Markdown 可复用的单元格。"""
        key = "加速度计零偏不稳定性_BI" if sensor == "acc" else "陀螺仪零偏不稳定性_BI"
        entry = self.report_data.get(key, {}).get(axis, {})
        if not entry:
            old_bs_key = (
                "加速度计零偏稳定性_10s平滑"
                if sensor == "acc" else "陀螺仪零偏稳定性_10s平滑"
            )
            old_bs = self.report_data.get(old_bs_key, {}).get(axis, "N/A")
            return {
                "rw": "N/A", "bi": "N/A", "min_ref": "N/A",
                "bs": old_bs, "method": "N/A",
            }

        def fmt(value, unit):
            try:
                return "N/A" if value is None or not np.isfinite(value) else f"{value:.6e} {unit}"
            except (TypeError, ValueError):
                return "N/A"

        rw_result = entry.get("rw", {})
        bi_result = entry.get("bi", {})
        min_ref_result = entry.get("min_ref", {})
        rw_units = entry.get("rw_units", {})
        bi_units = entry.get("bi_units", {})
        min_ref_units = entry.get("min_ref_units", {})
        bs_units = entry.get("bs_units", {})
        if sensor == "acc":
            rw = "<br>".join([
                fmt(rw_units.get("m/s/√s"), "m/s/√s"),
                fmt(rw_units.get("m/s/√h"), "m/s/√h"),
                fmt(rw_units.get("μg·√s"), "μg·√s"),
                "单边白噪声ASD等效：" +
                fmt(rw_units.get("单边ASD_μg/√Hz"), "μg/√Hz"),
            ])
            bi = "<br>".join([
                fmt(bi_units.get("m/s²"), "m/s²"),
                fmt(bi_units.get("g"), "g"),
                fmt(bi_units.get("mg"), "mg"),
                fmt(bi_units.get("μg"), "μg"),
            ])
            min_ref = "<br>".join([
                fmt(min_ref_units.get("m/s²"), "m/s²"),
                fmt(min_ref_units.get("g"), "g"),
                fmt(min_ref_units.get("mg"), "mg"),
                fmt(min_ref_units.get("μg"), "μg"),
            ])
            bs = "<br>".join([
                fmt(bs_units.get("m/s²"), "m/s²"),
                fmt(bs_units.get("mg"), "mg"),
                fmt(bs_units.get("μg"), "μg"),
            ])
        else:
            rw = "<br>".join([
                fmt(rw_units.get("°/√s"), "°/√s"),
                fmt(rw_units.get("°/√h"), "°/√h"),
                fmt(rw_units.get("rad/√s"), "rad/√s"),
                fmt(rw_units.get("rad/√h"), "rad/√h"),
                "单边白噪声ASD等效：" +
                fmt(rw_units.get("单边ASD_(°/s)/√Hz"), "(°/s)/√Hz"),
            ])
            bi = "<br>".join([
                fmt(bi_units.get("°/s"), "°/s"),
                fmt(bi_units.get("°/h"), "°/h"),
                fmt(bi_units.get("rad/s"), "rad/s"),
                fmt(bi_units.get("rad/h"), "rad/h"),
            ])
            min_ref = "<br>".join([
                fmt(min_ref_units.get("°/s"), "°/s"),
                fmt(min_ref_units.get("°/h"), "°/h"),
                fmt(min_ref_units.get("rad/s"), "rad/s"),
                fmt(min_ref_units.get("rad/h"), "rad/h"),
            ])
            bs = "<br>".join([
                fmt(bs_units.get("°/s"), "°/s"),
                fmt(bs_units.get("°/h"), "°/h"),
                fmt(bs_units.get("rad/h"), "rad/h"),
            ])

        method = self.report_data.get("Allan分析配置", "未记录")
        if rw_result.get("valid"):
            rw_cell = f"有效（拟合斜率={rw_result.get('slope', np.nan):.3f}）<br>{rw}"
        else:
            rw_cell = f"N/A<br>{rw_result.get('reason', '无可靠白噪声拟合区')}"

        if bi_result.get("valid"):
            if bi_result.get("low_edge_limited") and bi_result.get("high_edge_limited"):
                bi_status = "两端有效候选边界受限（复核左边界原因并延长数据）"
            elif bi_result.get("low_edge_limited"):
                bi_status = "位于有效候选范围左边界（原因见诊断）"
            elif bi_result.get("high_edge_limited"):
                bi_status = "受记录长度/支持项上界限制（需更长数据）"
            else:
                bi_status = "平台质量判据通过"
            bi_diag = (
                f"平台tau≈{bi_result.get('platform_tau', np.nan):.6g} s；"
                f"斜率={bi_result.get('fit_slope', np.nan):.3f}；"
                f"最小n_terms={bi_result.get('fit_min_n_terms', np.nan):.0f}"
            )
            bi_cell = (
                f"{bi_status}<br>{bi}<br>{bi_diag}<br>"
                "规格可比性未确认；需另行匹配厂家测试条件与定义"
            )
        else:
            bi_cell = f"N/A<br>{bi_result.get('reason', '未找到可信平台')}"

        if min_ref_result.get("available"):
            slope = min_ref_result.get("local_slope", np.nan)
            slope_text = f"{slope:.3f}" if np.isfinite(slope) else "N/A"
            edge_text = "；候选范围边界" if min_ref_result.get("edge_limited") else ""
            min_ref_cell = (
                f"参考值，非正式BI<br>{min_ref}<br>"
                f"tau={min_ref_result.get('tau', np.nan):.6g} s；"
                f"n_terms={min_ref_result.get('n_terms', np.nan):.0f}；"
                f"局部斜率={slope_text}{edge_text}<br>"
                "不用于规格合格判定"
            )
        else:
            min_ref_cell = f"N/A<br>{min_ref_result.get('reason', '受约束候选点不足')}"
        return {
            "rw": rw_cell,
            "bi": bi_cell,
            "min_ref": min_ref_cell,
            "bs": bs,
            "method": method,
        }

    # ----------------------------------------------------------------
    # 数据完整性检测图（仅 huace 二进制数据；CSV 不适用）
    # ----------------------------------------------------------------
    def plot_data_integrity(self):
        """绘制数据完整性检测图并生成结论。

        审计口径（均基于原始全量文件，不受掐头去尾影响）：
        1. CRC32 校验：zlib CRC-32 覆盖 frame[2:50]（算法经全量实测验证），
           校验失败的帧计入坏帧，不进入分析；
        2. 坏帧：同步头异常帧 + CRC32 拒绝帧 + 其他解包失败帧 + 尾部不完整字节；
        3. 丢帧：相邻设备时间戳步长 > 1.5×中位步长记为一处缺口，
           推定缺失 floor(step/dt)-1 帧（±1 帧取整不确定性）；
        4. 采样率符合性：时间戳推定频率与用户配置频率的相对偏差，
           ≤0.1% 判"符合"（本项目工程判据，非协议规定）。
        """
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            src = getattr(self, "_integrity_source", None)
            if not src or "ts_unwrapped" not in src:
                self.report_data["数据完整性图"] = (
                    "仅 huace 二进制数据支持完整性检测（CSV 数据不适用）"
                )
                return None

            ts = np.asarray(src["ts_unwrapped"], dtype=np.float64)
            quality = src.get("quality", {})
            stats = src.get("stats", {})

            total_slots = int(quality.get("binary_total_slots", 0))
            valid_frames = int(quality.get("binary_valid_frames", 0))
            bad_sync = int(quality.get("binary_bad_sync_frames", 0))
            crc_rejected = int(quality.get("binary_crc_rejected_frames", 0))
            rejected = int(quality.get("binary_rejected_frames", 0))
            trailing = int(quality.get("binary_trailing_bytes", 0))

            steps = np.diff(ts)
            t_rel = (ts - ts[0]) / 1000.0  # 秒
            dt = float(stats.get("median_step_ms", np.median(steps)))
            inferred_hz = float(stats.get("inferred_rate_hz", 1000.0 / dt if dt else float("nan")))
            gap_events = int(stats.get("gap_events", 0))
            missing_frames = int(stats.get("missing_frames", 0))
            gap_details = stats.get("gap_details", [])
            gap_truncated = bool(stats.get("gap_details_truncated", False))
            duplicate_pairs = int(stats.get("duplicate_pairs", 0))

            # 采样率符合性（工程判据：相对偏差 ≤ 0.1% 判为符合）
            if np.isfinite(inferred_hz) and self.sample_rate > 0:
                rel_dev_pct = abs(inferred_hz - self.sample_rate) / self.sample_rate * 100.0
                rate_conform = rel_dev_pct <= 0.1
            else:
                rel_dev_pct = float("nan")
                rate_conform = False

            # 异常步进掩码（丢帧或重复）
            anomaly_mask = (steps > dt * 1.5) | (steps < dt * 0.5)

            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            fig.suptitle("数据完整性检测（基于原始全量文件）", fontsize=16, fontweight="bold")

            # ── (0,0) 时间戳步长曲线 ──
            ax = axes[0, 0]
            stride = max(1, len(steps) // 200000)  # 抽样绘制正常步长
            ax.plot(t_rel[1:][::stride], steps[::stride], color="#1f77b4",
                    linewidth=0.6, alpha=0.8, label=f"帧间时间戳步长（{stride}点抽样）")
            if np.any(anomaly_mask):
                ax.plot(t_rel[1:][anomaly_mask], steps[anomaly_mask], "r.",
                        markersize=4, alpha=0.7, label=f"异常步进（{int(np.count_nonzero(anomaly_mask))} 处）")
            ax.axhline(dt, color="#2ca02c", linestyle="--", linewidth=1.2,
                       label=f"基准步长 {dt:.3f} ms")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("步长 (ms)")
            ax.set_title("帧间时间戳步长", fontweight="bold")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            # ── (0,1) 步长分布直方图 ──
            ax = axes[0, 1]
            step_range = (max(dt - 1.0, 0), dt + 1.0) if np.all(~anomaly_mask) else (0, float(np.max(steps)))
            ax.hist(steps, bins=80, range=step_range, color="#1f77b4",
                    edgecolor="black", linewidth=0.3)
            ax.axvline(dt, color="#2ca02c", linestyle="--", linewidth=1.2,
                       label=f"中位步长 {dt:.3f} ms")
            ax.set_yscale("log")
            ax.set_xlabel("步长 (ms)")
            ax.set_ylabel("帧数（对数）")
            ax.set_title("时间戳步长分布", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── (1,0) 累计接收帧数 vs 设备时间 ──
            ax = axes[1, 0]
            cum = np.arange(1, len(ts) + 1)
            stride2 = max(1, len(ts) // 300000)
            ax.plot(t_rel[::stride2], cum[::stride2], color="#1f77b4",
                    linewidth=1.0, label="累计接收帧数")
            if dt > 0:
                ax.plot(t_rel[::stride2], t_rel[::stride2] * 1000.0 / dt, color="#2ca02c",
                        linestyle="--", linewidth=1.0,
                        label=f"理想连续线（{1000.0/dt:.2f} Hz）")
            if gap_details:
                gap_t = [(g["ts_before_ms"] - ts[0]) / 1000.0 for g in gap_details[:200]]
                ax.vlines(gap_t, 0, len(ts), colors="red", linestyles=":",
                          linewidth=0.8, alpha=0.8,
                          label=f"丢帧位置（前 {min(len(gap_details), 200)} 处）")
            ax.set_xlabel("设备时间 (s)")
            ax.set_ylabel("累计帧数")
            ax.set_title("累计帧数增长与丢帧位置", fontweight="bold")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            # ── (1,1) 结论摘要面板 ──
            ax = axes[1, 1]
            ax.axis("off")
            bad_total = bad_sync + crc_rejected + max(rejected - crc_rejected, 0)
            if bad_total == 0 and gap_events == 0 and rate_conform:
                verdict, verdict_color = "√ 数据完整：无坏帧、无丢帧、采样率符合", "#2ca02c"
            elif bad_total == 0 and gap_events == 0 and not rate_conform:
                verdict, verdict_color = "! 无坏帧、无丢帧；采样率与配置偏差超阈值，请核对", "#e67e22"
            else:
                verdict, verdict_color = "! 检测到数据完整性问题，详见下列统计", "#e53e3e"

            if np.isfinite(rel_dev_pct):
                rate_line = (f"采样率：配置 {self.sample_rate:g} Hz；时间戳推定 {inferred_hz:.2f} Hz；"
                             f"相对偏差 {rel_dev_pct:.3f}% → {'符合' if rate_conform else '不符合'}"
                             f"（判据：偏差≤0.1%，本项目工程判据）")
            else:
                rate_line = "采样率：无法由时间戳推定"

            if gap_events == 0:
                gap_line = f"丢帧：未检测到丢帧（{int(np.count_nonzero(anomaly_mask))} 处异常步进）" \
                           if np.any(anomaly_mask) else "丢帧：未检测到丢帧（时间戳步长全程均匀）"
            else:
                ratio = stats.get("missing_ratio", float("nan"))
                ratio_txt = f"，占推定完整时长 {ratio*100:.4f}%" if np.isfinite(ratio) else ""
                gap_line = f"丢帧：检测到 {gap_events} 处缺口，推定丢失 {missing_frames} 帧{ratio_txt}"
                if crc_rejected > 0:
                    gap_line += f"（口径说明：其中包含 CRC32 拒绝的 {crc_rejected} 帧在数据流中留下的空洞，与坏帧统计不叠加计数）"

            lines = [
                ("数据完整性检测结果", "title"),
                (f"数据文件: {os.path.basename(self.file_path)}", "info"),
                ("CRC32 校验: 已启用（zlib CRC-32，覆盖 id+length+payload，算法经实测验证）", "info"),
                (f"总帧数（文件槽位）: {total_slots:,}；尾部不完整字节: {trailing}", "info"),
                (f"有效解析帧: {valid_frames:,}", "info"),
                (f"坏帧: 同步头异常 {bad_sync} / CRC32 拒绝 {crc_rejected} / 其他 {max(rejected - crc_rejected, 0)}",
                 "warn" if bad_total else "info"),
                (gap_line, "warn" if gap_events else "info"),
                (f"重复帧（时间戳相同）: {duplicate_pairs} 对", "info"),
                (f"采样间隔: 中位 {dt:.3f} ms（最小 {float(np.min(steps)):.3f} / 最大 {float(np.max(steps)):.3f}）", "info"),
                (rate_line, "info" if rate_conform else "warn"),
                ("", "info"),
                (verdict, "verdict"),
            ]
            y = 0.98
            for text, style in lines:
                if style == "title":
                    ax.text(0.02, y, text, fontsize=13, fontweight="bold",
                            color="#2d3748", transform=ax.transAxes, va="top")
                    y -= 0.075
                elif style == "verdict":
                    ax.text(0.02, y, text, fontsize=12.5, fontweight="bold",
                            color=verdict_color, transform=ax.transAxes, va="top")
                    y -= 0.06
                else:
                    color = "#e53e3e" if style == "warn" else "#4a5568"
                    ax.text(0.02, y, text, fontsize=10, color=color,
                            transform=ax.transAxes, va="top")
                    y -= 0.055

            # 丢帧明细（若有，最多列 30 条）
            if gap_details:
                y -= 0.01
                ax.text(0.02, y, "丢帧明细（帧号/时间戳区间/缺失数）：", fontsize=9.5,
                        fontweight="bold", color="#e53e3e", transform=ax.transAxes, va="top")
                y -= 0.045
                for g in gap_details[:30]:
                    txt = (f"第 {g['frame_index']:,} 帧后: timer {g['ts_before_ms']:,} → "
                           f"{g['ts_after_ms']:,} ms，步长 {g['step_ms']} ms，"
                           f"推定缺失 {g['missing_frames']} 帧（{g['missing_duration_s']:.3f} s）")
                    ax.text(0.04, y, txt, fontsize=8.2, color="#4a5568",
                            transform=ax.transAxes, va="top", family="sans-serif")
                    y -= 0.035
                shown = min(len(gap_details), 30)
                remaining = gap_events - shown
                note = f"（以上为前 {shown} 条"
                if gap_truncated or remaining > 0:
                    note += f"；共 {gap_events} 处，完整明细见报告文字"
                note += "）"
                ax.text(0.04, y, note, fontsize=8.2, color="#718096",
                        transform=ax.transAxes, va="top")
                ax.text(0.02, 0.01,
                        "推定缺失帧数 = floor(步长/基准步长) - 1，存在 ±1 帧取整不确定性",
                        fontsize=7.5, color="#a0aec0", transform=ax.transAxes, va="bottom")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(self.save_dir, "08_数据完整性检测图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            self.report_data["数据完整性图"] = save_path

            # ── 写入报告文字结论（自动出现在 HTML/MD 基本信息）──
            self.report_data["完整性检测总帧数"] = (
                f"{total_slots:,} 槽位 / 有效 {valid_frames:,} 帧 / 尾部余 {trailing} 字节"
            )
            self.report_data["坏帧统计"] = (
                f"同步头异常 {bad_sync}，CRC32 拒绝 {crc_rejected}，"
                f"其他解包失败 {max(rejected - crc_rejected, 0)}"
            )
            self.report_data["丢帧统计"] = gap_line
            if np.isfinite(rel_dev_pct):
                self.report_data["采样间隔统计"] = (
                    f"中位 {dt:.3f} ms（最小 {float(np.min(steps)):.3f} / 最大 {float(np.max(steps)):.3f}），"
                    f"对应 {1000.0/dt:.2f} Hz"
                )
                self.report_data["采样率符合性"] = (
                    f"配置 {self.sample_rate:g} Hz vs 推定 {inferred_hz:.2f} Hz，"
                    f"偏差 {rel_dev_pct:.3f}%，{'符合' if rate_conform else '不符合'}"
                    "（判据 ≤0.1%，工程判据）"
                )
            if gap_details:
                detail_lines = [
                    f"第 {g['frame_index']:,} 帧后 timer {g['ts_before_ms']:,}→{g['ts_after_ms']:,} ms，"
                    f"推定缺失 {g['missing_frames']} 帧（{g['missing_duration_s']:.3f} s）"
                    for g in gap_details[:50]
                ]
                suffix = f"（共 {gap_events} 处，仅列前 {min(len(gap_details), 50)} 条）" \
                    if gap_events > 50 else ""
                self.report_data["丢帧明细"] = "；".join(detail_lines) + suffix
            self.report_data["数据完整性结论"] = verdict
            return save_path
        except Exception as e:
            print(f"数据完整性检测图生成失败: {e}")
            import traceback as _tb
            _tb.print_exc()
            self.report_data["数据完整性图"] = f"生成失败: {e}"
            return None

    def generate_html_report(self):
        """生成HTML报告"""
        try:
            html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMU数据分析报告</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #1a202c;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
        }
        .info-box {
            background: #edf2f7;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
        }
        .info-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: white;
            border-radius: 6px;
        }
        .info-label {
            font-weight: bold;
            color: #4a5568;
        }
        .info-value {
            color: #2d3748;
        }
        .image-section {
            margin-top: 30px;
        }
        .image-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 15px;
        }
        .result-image {
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .stats-table th, .stats-table td {
            border: 1px solid #e2e8f0;
            padding: 12px;
            text-align: left;
        }
        .stats-table th {
            background: #667eea;
            color: white;
        }
        .stats-table tr:nth-child(even) {
            background: #f7fafc;
        }
        .warning {
            color: #e53e3e;
            background: #fff5f5;
            padding: 10px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IMU 数据分析报告</h1>
        <p class="subtitle">Powered by IMU Analysis Toolkit</p>
"""

            html += "\n        <div class='info-box'>\n            <div class='info-title'>数据基本信息</div>\n            <div class='info-grid'>\n"

            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位",
                "Allan曲线数据", "Allan参数汇总",
                "Allan结果", "时间序列图", "统计分布图", "Allan方差图", "Allan偏差图",
                "PSD图", "相关性图", "漂移分析图", "落点半径对比图", "落点半径对比数据",
                "数据完整性图",
            }
            for key, value in self.report_data.items():
                if key not in hidden_report_keys:
                    html += f"                <div class='info-item'><span class='info-label'>{key}:</span><span class='info-value'>{value}</span></div>\n"

            html += "            </div>\n        </div>\n"

            if "统计摘要" in self.report_data and self.report_data["统计摘要"]:
                html += "\n        <div class='info-box'>\n            <div class='info-title'>统计摘要</div>\n            <table class='stats-table'>\n                <tr><th>轴</th><th>均值</th><th>标准差</th><th>RMS</th></tr>\n"
                for axis, stats in self.report_data["统计摘要"].items():
                    html += f"                <tr><td>{axis}</td><td>{stats['均值']}</td><td>{stats['标准差']}</td><td>{stats['RMS']}</td></tr>\n"
                html += "            </table>\n        </div>\n"

            has_allan_results = bool(
                self.report_data.get("加速度计零偏不稳定性_BI") or
                self.report_data.get("陀螺仪零偏不稳定性_BI") or
                self.report_data.get("加速度计零偏稳定性_10s平滑") or
                self.report_data.get("陀螺仪零偏稳定性_10s平滑")
            )
            if has_allan_results:
                html += "\n        <div class='info-box'>\n            <div class='info-title'>Allan 偏差与零偏结果</div>\n"
                html += "            <table class='stats-table'>\n"
                html += "                <tr><th>指标</th><th>传感器</th><th>方法/口径</th><th>单位与数值（X）</th><th>单位与数值（Y）</th><th>单位与数值（Z）</th></tr>\n"
                for sensor, label in [("gyro", "陀螺仪"), ("acc", "加速度计")]:
                    cells = [self._allan_report_cells(sensor, axis) for axis in ["x", "y", "z"]]
                    html += f"                <tr><td><b>正式零偏不稳定性 BI</b></td><td>{label}</td><td>{cells[0]['method']}<br>B=σ平台/0.66428；连续近零斜率平台<br>门槛为本项目工程判据</td>"
                    html += "".join(f"<td>{cells[j]['bi']}</td>" for j in range(3)) + "</tr>\n"
                    html += f"                <tr><td><b>最低 ADEV 等效参考</b><br><span style='color:#b83280;'>非正式 BI</span></td><td>{label}</td><td>tau≥1 s、n_terms≥20；σmin/0.66428<br>不用于规格判定</td>"
                    html += "".join(f"<td>{cells[j]['min_ref']}</td>" for j in range(3)) + "</tr>\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    html += f"                <tr><td><b>{rw_label}</b></td><td>{label}</td><td>{cells[0]['method']}<br>0.1～10 s斜率拟合</td>"
                    html += "".join(f"<td>{cells[j]['rw']}</td>" for j in range(3)) + "</tr>\n"
                    html += f"                <tr><td><b>10 s 分段均值标准差</b></td><td>{label}</td><td>非重叠分段；ddof=0</td>"
                    html += "".join(f"<td>{cells[j]['bs']}</td>" for j in range(3)) + "</tr>\n"
                html += "            </table>\n        </div>\n"

                # 指标定义说明
                html += """
        <div class='info-box'>
            <div class='info-title'>指标定义说明</div>
            <table class='stats-table' style='font-size:13px;'>
                <tr><th style='background:#4a5568;'>指标名称</th><th style='background:#4a5568;'>定义</th><th style='background:#4a5568;'>计算方法</th><th style='background:#4a5568;'>标准依据</th></tr>
                <tr><td><b>零偏不稳定性 (BI)</b></td>
                    <td>Allan 偏差曲线中的 flicker/pink rate-noise 平台系数</td>
                    <td>识别3个原曲线连续点：|拟合斜率|≤0.10、max|相邻斜率|≤0.15、log10残差RMS≤0.03；B = σ平台 / 0.66428；平台识别通过不等于规格可比，所有平台均需另行匹配厂家测试条件与定义，边界平台另有边界限制</td>
                    <td>噪声模型关系与工程判据；非通用 MEMS 合规声明</td></tr>
                <tr><td><b>最低 ADEV 等效参考</b></td>
                    <td>可信平台不明显时仍可报告的受约束工程参考；不是 BI</td>
                    <td>仅在 tau≥1 s 且 n_terms≥20 中取最低 ADEV，再除以 0.66428；同时报告 tau、n_terms、局部斜率和边界状态</td>
                    <td>本项目工程判据，非标准强制门槛；不用于规格合格判定</td></tr>
                <tr><td><b>10 s 分段均值标准差</b></td>
                    <td>10 秒非重叠窗口均值的总体标准差</td>
                    <td>按10 s分段，计算各段均值的 std(ddof=0)</td>
                    <td>工程统计量，条款适用性需另行核对</td></tr>
                <tr><td><b>角度随机游走 (ARW)</b></td>
                    <td>陀螺仪白噪声引起的角度积分随机游走</td>
                    <td>ADEV 在 0.1～10 s 区间进行 log-log 斜率拟合，外推至 1 s</td>
                    <td>Allan 噪声模型参考</td></tr>
                <tr><td><b>速度随机游走 (VRW)</b></td>
                    <td>加速度计白噪声引起的速度积分随机游走</td>
                    <td>ADEV 在 0.1～10 s 区间进行 log-log 斜率拟合，外推至 1 s</td>
                    <td>Allan 噪声模型参考</td></tr>
            </table>
        </div>
"""

            if "落点半径对比数据" in self.report_data and self.report_data["落点半径对比数据"]:
                html += "\n        <div class='info-box'>\n            <div class='info-title'>不同时间窗口1σ落点半径</div>\n            <table class='stats-table'>\n                <tr><th>时间窗口</th><th>1σ落点半径</th><th>有效窗口数</th></tr>\n"
                for row in self.report_data["落点半径对比数据"]:
                    html += f"                <tr><td>{row['时间窗口']}</td><td>{row['1σ落点半径']}</td><td>{row['有效窗口数']}</td></tr>\n"
                html += "            </table>\n        </div>\n"

            image_keys = [
                ("时间序列图", "1. 时间序列图"),
                ("统计分布图", "2. 统计分布分析图"),
                ("Allan方差图", "3. Allan偏差与零偏分析图"),
                ("PSD图", "4. 功率谱密度（PSD）分析图"),
                ("相关性图", "5. 相关性分析图"),
                ("数据完整性图", "6. 数据完整性检测图"),
                ("漂移分析图", "7. 长期零漂趋势图"),
                ("落点半径对比图", "8. 不同时间窗口1σ落点半径对比图")
            ]

            for key, title in image_keys:
                if key in self.report_data:
                    img_path = self.report_data[key]
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        html += f"\n        <div class='image-section'>\n            <div class='image-title'>{title}</div>\n            <img src='{img_name}' class='result-image' alt='{title}' />\n        </div>\n"
                    elif isinstance(img_path, str):
                        html += f"\n        <div class='image-section'>\n            <div class='image-title'>{title}</div>\n            <div class='warning'>{img_path}</div>\n        </div>\n"

            # Allan 偏差单位换算表（每次报告都附加）
            html += """
        <div class='image-section'>
            <div class='image-title'>Allan 偏差常用单位换算参考表</div>
            <table class='stats-table' style='font-family: monospace; font-size: 13px;'>
                <tr><th style='background:#4a5568;'>参数</th><th style='background:#4a5568;'>计算单位</th><th style='background:#4a5568;'>换算公式</th><th style='background:#4a5568;'>目标单位</th></tr>

                <tr style='background:#fef9e7;'><td rowspan='3'><b>加速度计 VRW 系数 N</b><br><span style='color:#666;'>(Velocity Random Walk)</span></td>
                    <td rowspan='3'>m/s/√s（即 (m/s²)·√s）</td>
                    <td>= N</td><td>m/s/√s</td></tr>
                <tr style='background:#fef9e7;'><td>= N × 60</td><td>m/s/√h</td></tr>
                <tr style='background:#fef9e7;'><td>= N × 1×10⁶ / 9.80665</td><td>μg·√s</td></tr>

                <tr style='background:#fff8dc;'><td rowspan='3'><b>加速度计 BI/等效参考/BS</b><br><span style='color:#666;'>(Bias / reference / 10 s segment statistic)</span></td>
                    <td rowspan='3'>m/s²</td>
                    <td>= 数值 / 9.80665</td><td>g</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × 1×10³ / 9.80665</td><td>mg</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × 1×10⁶ / 9.80665</td><td>μg</td></tr>
                <tr style='background:#fef9e7;'><td rowspan='3'><b>陀螺仪 ARW 系数 N</b><br><span style='color:#666;'>(Angle Random Walk)</span></td>
                    <td rowspan='3'>°/√s</td>
                    <td>= N × 60</td><td>°/√h</td></tr>
                <tr style='background:#fef9e7;'><td>= N × π / 180</td><td>rad/√s</td></tr>
                <tr style='background:#fef9e7;'><td>= N × π / 3</td><td>rad/√h</td></tr>

                <tr style='background:#eef6ff;'><td rowspan='2'><b>条件等效单边白噪声 ASD</b></td>
                    <td rowspan='2'>仅限白速率噪声、0..fs/2 单边 PSD 约定</td>
                    <td>= √2 × N</td><td>(输入速率单位)/√Hz</td></tr>
                <tr style='background:#eef6ff;'><td colspan='2'>由 ADEV-RW 推导；不是直接谱估计，也不自动等同厂家噪声密度</td></tr>

                <tr style='background:#fff8dc;'><td rowspan='3'><b>陀螺仪 BI/等效参考/BS</b><br><span style='color:#666;'>(Bias / reference / 10 s segment statistic)</span></td>
                    <td rowspan='3'>°/s</td>
                    <td>= 数值 × 3600</td><td>°/h</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × π / 180</td><td>rad/s</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × 20π</td><td>rad/h</td></tr>
            </table>
            <div style='margin-top:10px; padding:10px; background:#f0f4f8; border-left:4px solid #667eea; font-size:13px;'>
                <b>📖 说明：</b><br>
                1. <b>输入同量纲</b>：ADEV 的单位由输入速率序列决定，图轴不因结果换算而改变。<br>
                2. <b>BI</b>：仅对识别出的近零斜率平台计算，关系为 B = σ平台 / 0.66428。<br>
                3. <b>最低 ADEV 等效参考</b>：只是受约束的替代参考，不是正式 BI，不用于规格判定。<br>
                4. <b>BS</b>：本程序显示的是10 s非重叠分段均值标准差（ddof=0）这一工程统计量。<br>
                5. <b>单边 ASD 条件等效</b>：√2×N 仅适用于白速率噪声和0..fs/2单边PSD约定；不是直接谱估计，也不自动等同厂家噪声密度。<br>
                6. <b>1 g</b> = 9.80665 m/s²；1 mg = 0.001 g；1 μg = 0.000001 g。
            </div>
        </div>
"""

            html += """
    </div>
</body>
</html>
"""

            html_path = os.path.join(self.save_dir, "IMU_Analysis_Report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            return html_path
        except Exception as e:
            print(f"HTML报告生成失败: {e}")
            return None

    def generate_md_report(self):
        """生成Markdown报告"""
        try:
            md = "# IMU 数据分析报告\n\n"
            md += "---\n\n## 数据基本信息\n\n"

            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位",
                "Allan曲线数据", "Allan参数汇总",
                "Allan结果", "时间序列图", "统计分布图", "Allan方差图", "Allan偏差图",
                "PSD图", "相关性图", "漂移分析图", "落点半径对比图", "落点半径对比数据",
                "数据完整性图",
            }
            for key, value in self.report_data.items():
                if key not in hidden_report_keys:
                    md += f"- **{key}**: {value}\n"

            if "统计摘要" in self.report_data and self.report_data["统计摘要"]:
                md += "\n## 统计摘要\n\n"
                md += "| 轴 | 均值 | 标准差 | RMS |\n"
                md += "|-----|-----|-----|-----|\n"
                for axis, stats in self.report_data["统计摘要"].items():
                    md += f"| {axis} | {stats['均值']} | {stats['标准差']} | {stats['RMS']} |\n"

            has_allan_results = bool(
                self.report_data.get("加速度计零偏不稳定性_BI") or
                self.report_data.get("陀螺仪零偏不稳定性_BI") or
                self.report_data.get("加速度计零偏稳定性_10s平滑") or
                self.report_data.get("陀螺仪零偏稳定性_10s平滑")
            )
            if has_allan_results:
                md += "\n## Allan 偏差与零偏结果\n\n"
                md += "| 指标 | 传感器 | 方法/口径 | X轴 | Y轴 | Z轴 |\n"
                md += "|------|------|------|-----|-----|-----|\n"
                for sensor, label in [("gyro", "陀螺仪"), ("acc", "加速度计")]:
                    cells = [self._allan_report_cells(sensor, axis) for axis in ["x", "y", "z"]]
                    md += f"| **正式零偏不稳定性 BI** | {label} | {cells[0]['method']}；B=σ平台/0.66428；连续近零斜率平台；门槛为本项目工程判据 |"
                    for cell in cells:
                        md += f" {cell['bi']} |"
                    md += "\n"
                    md += f"| **最低 ADEV 等效参考（非正式 BI）** | {label} | tau≥1 s、n_terms≥20；σmin/0.66428；不用于规格判定 |"
                    for cell in cells:
                        md += f" {cell['min_ref']} |"
                    md += "\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    md += f"| **{rw_label}** | {label} | {cells[0]['method']}；0.1～10 s斜率拟合 |"
                    for cell in cells:
                        md += f" {cell['rw']} |"
                    md += "\n"
                    md += f"| **10 s 分段均值标准差** | {label} | 非重叠分段；ddof=0 |"
                    for cell in cells:
                        md += f" {cell['bs']} |"
                    md += "\n"

                md += """
### 指标定义说明

| 指标名称 | 定义 | 计算方法 | 标准依据 |
|----------|------|----------|----------|
| **零偏不稳定性 (BI)** | Allan 偏差中的 flicker/pink rate-noise 平台系数 | 3个原曲线连续点：|拟合斜率|≤0.10、max|相邻斜率|≤0.15、log10残差RMS≤0.03；B=σ平台/0.66428；平台识别通过不等于规格可比，所有平台均需匹配厂家测试条件，边界平台另有边界限制 | 噪声模型关系与本项目工程判据；门槛非标准强制值 |
| **最低 ADEV 等效参考** | 无可信平台时仍可报告的受约束工程参考；不是 BI | tau≥1 s且n_terms≥20的最低ADEV / 0.66428；报告tau、n_terms、局部斜率和边界状态 | 本项目工程判据，非标准强制门槛；不用于规格判定 |
| **10 s 分段均值标准差** | 10 秒非重叠窗口均值的总体标准差 | 分段均值后计算 std(ddof=0) | 工程统计量，条款适用性需另行核对 |
| **角度随机游走 (ARW)** | 陀螺仪白噪声引起的角度积分随机游走 | ADEV 在0.1～10 s区间拟合斜率并外推至1 s | Allan 噪声模型参考 |
| **速度随机游走 (VRW)** | 加速度计白噪声引起的速度积分随机游走 | ADEV 在0.1～10 s区间拟合斜率并外推至1 s | Allan 噪声模型参考 |
"""

            if "落点半径对比数据" in self.report_data and self.report_data["落点半径对比数据"]:
                md += "\n## 不同时间窗口1σ落点半径\n\n"
                md += "| 时间窗口 | 1σ落点半径 | 有效窗口数 |\n"
                md += "|-------|-------|-------|\n"
                for row in self.report_data["落点半径对比数据"]:
                    md += f"| {row['时间窗口']} | {row['1σ落点半径']} | {row['有效窗口数']} |\n"

            md += "\n## 分析图表\n\n"

            image_keys = [
                ("时间序列图", "1. 时间序列图"),
                ("统计分布图", "2. 统计分布分析图"),
                ("Allan方差图", "3. Allan偏差与零偏分析图"),
                ("PSD图", "4. 功率谱密度（PSD）分析图"),
                ("相关性图", "5. 相关性分析图"),
                ("数据完整性图", "6. 数据完整性检测图"),
                ("漂移分析图", "7. 长期零漂趋势图"),
                ("落点半径对比图", "8. 不同时间窗口1σ落点半径对比图")
            ]

            for key, title in image_keys:
                if key in self.report_data:
                    img_path = self.report_data[key]
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        md += f"\n### {title}\n\n"
                        md += f"![{title}]({img_name})\n"
                    elif isinstance(img_path, str):
                        md += f"\n### {title}\n\n"
                        md += f"*{img_path}*\n"

            # Allan 偏差单位换算表
            md += """
### Allan 偏差常用单位换算参考表

| 参数 | 计算单位 | 换算公式 | 目标单位 |
|---|---|---|---|
| **加速度计 VRW 系数 N** (Velocity Random Walk) | m/s/√s（即 (m/s²)·√s） | = N | m/s/√s |
|  |  | = N × 60 | m/s/√h |
|  |  | = N × 1×10⁶ / 9.80665 | μg·√s |
| **加速度计 BI/等效参考/BS** | m/s² | = 数值 / 9.80665 | g |
|  |  | = 数值 × 1×10³ / 9.80665 | mg |
|  |  | = 数值 × 1×10⁶ / 9.80665 | μg |
| **陀螺仪 ARW 系数 N** (Angle Random Walk) | °/√s | = N × 60 | °/√h |
|  |  | = N × π / 180 | rad/√s |
|  |  | = N × π / 3 | rad/√h |
| **条件等效单边白噪声 ASD** | 仅限白速率噪声和0..fs/2单边PSD约定 | = √2 × N | (输入速率单位)/√Hz |
|  |  | 由 ADEV-RW 推导，非直接谱估计 | 不自动等同厂家噪声密度 |
| **陀螺仪 BI/等效参考/BS** | °/s | = 数值 × 3600 | °/h |
|  |  | = 数值 × π / 180 | rad/s |
|  |  | = 数值 × 20π | rad/h |

**📖 说明：**

1. **输入同量纲**：ADEV 的单位由输入速率序列决定，图轴保持原坐标单位。
2. **BI**：仅对连续近零斜率平台计算，B = σ平台 / 0.66428；无平台时为 N/A。
3. **最低 ADEV 等效参考**：仅在 tau≥1 s、n_terms≥20 的候选中选取，它不是正式 BI，不用于规格判定。
4. **BS**：本程序显示10 s非重叠分段均值标准差（ddof=0）这一工程统计量。
5. **单边 ASD 条件等效**：√2×N 仅适用于白速率噪声和0..fs/2单边PSD约定；不是直接谱估计，也不自动等同厂家噪声密度。
6. **1 g** = 9.80665 m/s²；1 mg = 0.001 g；1 μg = 0.000001 g。
"""

            md_path = os.path.join(self.save_dir, "IMU_Analysis_Report.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

            return md_path
        except Exception as e:
            print(f"Markdown报告生成失败: {e}")
            return None

    # ----------------------------------------------------------------
    # 完整分析流水线
    # ----------------------------------------------------------------
    def run_full_analysis(self, progress_callback=None):
        """运行完整分析"""
        steps = [
            ("计算基本统计", self.calculate_basic_stats),
            ("计算10 s分段均值标准差", self.calculate_bias_stability_10s),
            ("绘制时间序列图", self.plot_time_series),
            ("绘制统计分布图", self.plot_distribution),
            ("绘制Allan偏差图", self.plot_allan_variance),
            ("绘制PSD图", self.plot_psd),
            ("绘制相关性分析图", self.plot_correlation),
            ("绘制数据完整性检测图", self.plot_data_integrity),
            ("绘制漂移分析图", self.plot_drift),
            ("绘制落点半径对比图", self.plot_drift_radius_comparison),
            ("生成HTML报告", self.generate_html_report),
            ("生成Markdown报告", self.generate_md_report)
        ]

        total_steps = len(steps)
        for i, (step_name, func) in enumerate(steps):
            if progress_callback:
                progress_callback(i + 1, total_steps, step_name)
            try:
                func()
            except Exception as e:
                print(f"Error in {step_name}: {e}")
                import traceback as _tb
                _tb.print_exc()

        if progress_callback:
            progress_callback(total_steps, total_steps, "分析完成！")

        return self.save_dir


# ===================================================================
# GUI 应用程序
# ===================================================================
class IMUAnalysisApp:
    """主HMI界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("IMU 数据分析工具箱 v2.0 (huace AHRS 支持)")
        self.root.geometry("800x620")

        self.file_path = tk.StringVar()
        # 当前 huace AHRS 数据采集配置为 500 Hz（由设备时间戳步长 2 ms 实测推定）；
        # 其他档位仍可在下拉框中选择。
        self.sample_rate = tk.StringVar(value="500")
        self.trim_minutes = tk.StringVar(value="0")
        self.analyzing = False

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="IMU 数据分析工具箱",
                                font=("Microsoft YaHei", 20, "bold"))
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(main_frame,
                                  text="huace AHRS 一站式静态数据分析解决方案 (v2.0)",
                                  font=("Microsoft YaHei", 10),
                                  foreground="#718096")
        subtitle_label.pack(pady=(0, 25))

        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=8)

        ttk.Label(file_frame, text="数据文件:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.file_path, width=55,
                 font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=10)
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).pack(side=tk.LEFT)

        sr_frame = ttk.Frame(main_frame)
        sr_frame.pack(fill=tk.X, pady=8)

        ttk.Label(sr_frame, text="采样率 (Hz):",
                 font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(sr_frame, textvariable=self.sample_rate, width=12,
                               font=("Microsoft YaHei", 10), state="readonly",
                               values=["100", "200", "400", "500", "1000"])
        sr_combo.pack(side=tk.LEFT, padx=10)
        ttk.Label(sr_frame, text="（huace AHRS 当前采集为 500 Hz 档）",
                 font=("Microsoft YaHei", 9), foreground="#718096").pack(side=tk.LEFT)

        # 掐头去尾输入行
        trim_frame = ttk.Frame(main_frame)
        trim_frame.pack(fill=tk.X, pady=8)
        ttk.Label(trim_frame, text="掐头去尾 (分钟):",
                 font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        ttk.Entry(trim_frame, textvariable=self.trim_minutes, width=6,
                 font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=10)
        ttk.Label(trim_frame, text="（输入 0 或不填则不丢弃；例如输入 5，丢弃头尾各 5 分钟）",
                 font=("Microsoft YaHei", 9), foreground="#718096").pack(side=tk.LEFT)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)

        self.analyze_btn = ttk.Button(btn_frame, text="开始完整分析", command=self.start_analysis,
                                     width=25)
        self.analyze_btn.pack()

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=15)

        ttk.Label(progress_frame, text="分析进度:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate", length=500)
        self.progress.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(progress_frame, text="就绪", font=("Microsoft YaHei", 9))
        self.status_label.pack(side=tk.LEFT)

        info_frame = ttk.LabelFrame(main_frame, text="分析功能", padding="15")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        features = [
            "✓ 数据基本信息提取（采样率、时长、温度范围等）",
            "✓ 时间序列图（加速度计/陀螺仪三轴）",
            "✓ 统计分布分析（直方图+KDE）",
            "✓ Allan偏差（ADEV）与零偏平台分析",
            "✓ 功率谱密度（PSD）分析",
            "✓ 相关性分析（热图）",
            "✓ 长期零漂趋势分析",
            "✓ 不同时间窗口1σ落点半径对比",
            "✓ 数据完整性检测（CRC32校验、坏帧/丢帧审计、采样间隔周期核查，仅bin数据）",
            "✓ 生成HTML和Markdown报告",
            "✓ 支持 huace AHRS 二进制格式 (.bin，54字节/帧，A5 D5 同步字)"
        ]

        for feature in features:
            ttk.Label(info_frame, text=feature, font=("Microsoft YaHei", 10)).pack(anchor="w", pady=3)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="选择IMU数据文件",
            filetypes=[("所有支持的文件", "*.csv;*.bin"),
                      ("CSV文件", "*.csv"),
                      ("二进制文件", "*.bin"),
                      ("所有文件", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def update_progress(self, current, total, message):
        self.progress["value"] = (current / total) * 100
        self.status_label["text"] = message
        self.root.update_idletasks()

    def start_analysis(self):
        if not self.file_path.get():
            messagebox.showerror("错误", "请先选择数据文件！")
            return

        if self.analyzing:
            return

        # 验证采样率
        try:
            sr_str = self.sample_rate.get()
            sr_val = float(sr_str)
            if sr_val <= 0:
                raise ValueError("采样率必须大于0")
            if sr_val not in {100, 200, 400, 500, 1000}:
                # 允许但提醒用户
                pass
        except Exception as e:
            messagebox.showerror("错误", f"采样率输入错误: {e}\n请输入有效的数字（如200）")
            return

        # 读取掐头去尾值
        try:
            trim_val = float(self.trim_minutes.get().strip() or "0")
            if trim_val < 0:
                trim_val = 0
        except Exception:
            trim_val = 0

        self.analyzing = True
        self.analyze_btn["state"] = "disabled"

        def analyze_thread():
            save_dir = None
            error_msg = None
            try:
                analyzer = IMUDataAnalyzer(self.file_path.get(), sr_val, trim_minutes=trim_val)
                save_dir = analyzer.run_full_analysis(self.update_progress)
            except Exception as e:
                error_msg = str(e)
            finally:
                self.analyzing = False
                def update_gui():
                    self.analyze_btn.config(state="normal")
                    if error_msg is not None:
                        messagebox.showerror("错误", f"分析过程中发生错误:\n{error_msg}")
                    elif save_dir is not None:
                        messagebox.showinfo("成功",
                            f"分析完成！\n\n结果保存在:\n{save_dir}\n\n包含:\n- 分析图片\n- HTML报告\n- Markdown报告")
                self.root.after(0, update_gui)

        threading.Thread(target=analyze_thread, daemon=True).start()


# ===================================================================
# 主入口
# ===================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = IMUAnalysisApp(root)
    root.mainloop()
