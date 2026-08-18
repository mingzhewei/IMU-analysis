# 整合的IMU分析工具箱 — 武汉元生 yis321S 专用版
import os, sys, datetime

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
import matplotlib.font_manager as fm
import chardet
import allantools as at
from scipy.signal import welch, detrend
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import textwrap

# ---- 中文字体支持 ----
# 按优先级排序的中文字体关键字（避免选到 SimSun-ExtG 这类只含扩展 CJK 的字体）
_CN_FONT_PRIORITY = [
    'microsoft yahei',   # 微软雅黑 - Windows 首选
    'simhei',            # 黑体 - Windows 备选
    'simsun',            # 宋体（注意：会排除 SimSun-ExtB/ExtG 等扩展子字体）
    'msyh',              # msyh.ttc 文件名
    'noto sans cjk sc',  # Noto Sans CJK SC
    'noto sans sc',      # Noto Sans SC
    'source han sans sc',# 思源黑体
    'wenquanyi',         # 文泉驿
    'pingfang sc',       # 苹方 - macOS
    'heiti sc',          # 黑体-简 - macOS
]

# 收集所有可用中文字体，按优先级排序后取第一个
_CN_FONT = None
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
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = _CN_FALLBACK_NAMES + ['DejaVu Sans']
else:
    print("[字体] 警告：未检测到任何中文字体，中文将显示为豆腐块")
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


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


TID_COUNTER_MIN = 1
TID_COUNTER_MAX = 60000

# Allan 曲线显示使用更密的对数 tau 网格，仅提高曲线的尺度分辨率。
# BI/RW 及最低 ADEV 参考的参数提取继续使用原有 octave 网格，
# 避免因 tau 加密改变“连续 3 点”等现有工程判据的实际对数跨度。
ALLAN_DISPLAY_POINTS_PER_DECADE = 15
ALLAN_DISPLAY_TAU_GRID = "nominal_15_points_per_decade_plus_octave_anchors"
ALLAN_ESTIMATION_TAU_GRID = "octave"
ALLAN_DISPLAY_MIN_NORMAL_TERMS = 20
ALLAN_DISPLAY_MIN_MEDIUM_TERMS = 5


def build_allan_display_taus(n_samples, sample_rate,
                             points_per_decade=ALLAN_DISPLAY_POINTS_PER_DECADE):
    """构建经整数平均因子 m 量化的对数 tau 网格。

    ``points_per_decade`` 是名义密度；短 tau 端因 m 只能取整数，
    四舍五入后的重复 m 会被删除。上限保守地取 ``floor(N/3)``，
    使经典非重叠 ADEV 的显示尾端保留至少2个支持项；
    这不声称是所有实现和输入类型下的绝对最大 tau。
    另外显式并入所有 octave（2 的整数次幂）锚点，便于与原有
    octave 曲线逐点对照。
    返回值为秒，可直接传入 ``allantools.adev(..., taus=...)``。
    """
    try:
        n = int(n_samples)
        fs = float(sample_rate)
        density = int(points_per_decade)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Allan tau 网格参数必须是有效数值") from exc
    if n < 3:
        raise ValueError("Allan tau 网格至少需要3个样本")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Allan tau 网格需要正的有限采样率")
    if density <= 0:
        raise ValueError("Allan tau 每十倍程点数必须为正整数")

    max_m = n // 3
    if max_m < 1:
        raise ValueError("Allan tau 网格没有可用的平均因子")
    if max_m == 1:
        return np.asarray([1.0 / fs], dtype=float)

    max_log10_m = float(np.log10(max_m))
    exponents = np.arange(
        0.0, max_log10_m + 0.5 / density, 1.0 / density, dtype=float
    )
    averaging_factors = np.rint(np.power(10.0, exponents)).astype(np.int64)
    averaging_factors = averaging_factors[
        (averaging_factors >= 1) & (averaging_factors <= max_m)
    ]
    octave_factors = np.asarray(
        [1 << exponent for exponent in range(int(np.floor(np.log2(max_m))) + 1)],
        dtype=np.int64,
    )
    # 显式纳入 octave 锚点和最长有效尺度；unique 同时完成
    # 排序和整数量化去重。
    averaging_factors = np.unique(
        np.concatenate((averaging_factors, octave_factors,
                        np.asarray([max_m], dtype=np.int64)))
    )
    return averaging_factors.astype(float) / fs


def classify_allan_support(n_terms):
    """按显示用 n_terms 门槛分类；仅决定绘图样式，不修改 ADEV。"""
    ns = np.asarray(n_terms, dtype=float)
    support = np.full(ns.shape, "unknown_n_terms", dtype=object)
    finite = np.isfinite(ns)
    support[finite & (ns >= ALLAN_DISPLAY_MIN_NORMAL_TERMS)] = (
        "normal_n_terms_ge_20"
    )
    support[
        finite & (ns >= ALLAN_DISPLAY_MIN_MEDIUM_TERMS) &
        (ns < ALLAN_DISPLAY_MIN_NORMAL_TERMS)
    ] = "limited_n_terms_5_to_19"
    support[finite & (ns < ALLAN_DISPLAY_MIN_MEDIUM_TERMS)] = (
        "low_n_terms_lt_5"
    )
    return support


def calculate_tid_continuity(tid_values, modulus=TID_COUNTER_MAX,
                             sample_rate=None):
    """按工程假设审计元申 TID 在 1..60000 循环下的相邻转移。

    输出只描述可观测到的计数器缺口，不将其解释为已证实的串口、
    采集器或设备丢帧。计数器无法识别文件首尾缺失、整数个循环的缺失，
    也无法仅凭大跳变区分复位、乱序和长缺口。1..60000 循环及逐样本
    加一目前是基于样本的工程假设，尚未以厂家协议确认。
    """
    result = {
        "available": False,
        "reason": "未检测到可审计的 TID",
        "observed_valid_values": 0,
        "transition_count": 0,
        "counter_gap_events": 0,
        "inferred_missing_positions": 0,
        "duplicate_pairs": 0,
        "wrap_count": 0,
        "invalid_values": 0,
        "ambiguous_transitions": 0,
        "expected_positions": 0,
        "missing_ratio": float("nan"),
        "received_ratio": float("nan"),
        "missing_duration_s": float("nan"),
        "observed_duration_s": float("nan"),
        "expected_duration_s": float("nan"),
        "ratio_is_partial_estimate": False,
        "modulus": int(modulus),
    }
    if tid_values is None:
        return result

    values = pd.to_numeric(pd.Series(tid_values), errors="coerce").to_numpy(dtype=float)
    valid = (np.isfinite(values) & (values == np.floor(values)) &
             (values >= TID_COUNTER_MIN) & (values <= int(modulus)))
    result["invalid_values"] = int(np.count_nonzero(~valid))
    result["observed_valid_values"] = int(np.count_nonzero(valid))
    if len(values) < 2 or result["observed_valid_values"] == 0:
        return result

    integer_values = np.zeros(values.shape, dtype=np.int64)
    integer_values[valid] = values[valid].astype(np.int64, copy=False)
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid):
        result["reason"] = "没有相邻的有效 TID 对，无法审计计数连续性"
        return result

    previous = integer_values[:-1][pair_valid]
    current = integer_values[1:][pair_valid]
    raw_delta = current - previous
    step = np.mod(raw_delta, int(modulus))

    # 超过半个周期的跳变不强行当作长缺口。
    ambiguous = step > (int(modulus) // 2)
    trusted = ~ambiguous
    gaps = trusted & (step > 1)
    inferred = int(np.sum(step[gaps] - 1)) if np.any(gaps) else 0
    result.update({
        "available": True,
        "reason": (
            "按 1..60000 循环且逐样本加一的工程假设审计；"
            "该语义尚未以厂家协议确认"
        ),
        "transition_count": int(np.count_nonzero(pair_valid)),
        "counter_gap_events": int(np.count_nonzero(gaps)),
        "inferred_missing_positions": inferred,
        "duplicate_pairs": int(np.count_nonzero(trusted & (step == 0))),
        "wrap_count": int(np.count_nonzero(trusted & (raw_delta < 0))),
        "ambiguous_transitions": int(np.count_nonzero(ambiguous)),
        "ratio_is_partial_estimate": bool(
            result["invalid_values"] or np.count_nonzero(ambiguous)
        ),
    })
    duplicate_pairs = result["duplicate_pairs"]
    known_observed = result["observed_valid_values"] - duplicate_pairs
    expected_positions = known_observed + inferred
    result["expected_positions"] = int(max(expected_positions, 0))
    if result["expected_positions"] > 0:
        result["missing_ratio"] = inferred / result["expected_positions"]
        result["received_ratio"] = known_observed / result["expected_positions"]
    try:
        fs = float(sample_rate)
    except (TypeError, ValueError):
        fs = float("nan")
    if np.isfinite(fs) and fs > 0 and result["expected_positions"] > 0:
        result["missing_duration_s"] = inferred / fs
        result["observed_duration_s"] = known_observed / fs
        result["expected_duration_s"] = result["expected_positions"] / fs
    return result


class IMUDataAnalyzer:
    """IMU数据分析核心类"""
    
    def __init__(self, file_path, sample_rate=200, trim_minutes=0):
        self.file_path = file_path
        self.sample_rate = float(sample_rate)  # 确保采样率是浮点数
        self.trim_minutes = float(trim_minutes)  # 掐头去尾分钟数，0 表示不截断
        self.save_dir = os.path.join(os.path.dirname(file_path), "analysis_results")
        self.report_data = {}
        self.source_tid_continuity = {}
        self.tid_continuity = {}
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.load_data()
    
    def load_data(self):
        """加载数据并提取基本信息 - 增强鲁棒性"""
        encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]
        
        self.df = None
        read_exception = None
        selected_encoding = None
        for enc in encodings:
            try:
                self.df = pd.read_csv(
                    self.file_path,
                    encoding=enc,
                    skipinitialspace=True,
                    low_memory=False,
                    on_bad_lines="error",
                )
                selected_encoding = enc
                print(f"成功使用编码 {enc} 加载数据")
                break
            except Exception as e:
                read_exception = e
                print(f"使用编码 {enc} 失败: {e}")
                continue
        
        if self.df is None:
            raise ValueError(
                "无法完整加载 CSV：程序不会静默跳过坏行，"
                f"请检查编码和列数。最后错误: {read_exception}"
            )
        source_row_count = len(self.df)
        self.report_data["CSV读取策略"] = (
            f"严格读取（on_bad_lines=error）；编码={selected_encoding}；"
            "不静默跳过坏行"
        )
        
        self.df = self.df.loc[:, ~self.df.columns.astype(str).str.match(r"^Unnamed")]
        self.df.columns = self.df.columns.astype(str).str.strip()
        self.df = self.df.loc[:, self.df.columns != ""]

        tid_column = next((col for col in self.df.columns
                           if str(col).strip().lower() == "tid"), None)
        if tid_column is not None:
            self.source_tid_continuity = calculate_tid_continuity(
                self.df[tid_column], sample_rate=self.sample_rate
            )
        
        # 列名映射：支持原始采集数据的中文列名自动转换为标准英文列名（兼容原始数据和处理后数据）
        col_mapping = {
            "X轴加速度": "acc_x",
            "Y轴加速度": "acc_y",
            "Z轴加速度": "acc_z",
            "X轴角速度": "gyro_x",
            "Y轴角速度": "gyro_y",
            "Z轴角速度": "gyro_z",
        }
        self.df = self.df.rename(columns=col_mapping)
        
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
        self.report_data["CSV数值清理"] = (
            f"原始 {source_row_count} 行；六轴可用数值行 {len(self.df)} 行；"
            f"移除 {source_row_count - len(self.df)} 行含非数值/空值的传感器记录"
        )
        
        if len(self.df) < 100:
            raise ValueError(f"有效数据点太少！仅 {len(self.df)} 个点")
        
        self._apply_trim()
        self._update_tid_continuity_audit()
        self.extract_metadata()

    def _update_tid_continuity_audit(self):
        """更新当前分析数据的 TID 连续性审计，不推断丢帧原因。"""
        tid_column = next((col for col in self.df.columns
                           if str(col).strip().lower() == "tid"), None)
        if tid_column is None:
            self.tid_continuity = calculate_tid_continuity(None)
            self.report_data["TID连续性审计"] = "未检测到 TID 列"
            return

        self.tid_continuity = calculate_tid_continuity(
            self.df[tid_column], sample_rate=self.sample_rate
        )
        stats = self.tid_continuity
        self.report_data["TID审计范围"] = "当前分析数据（数值清理和提头去尾后）"
        if not stats.get("available", False):
            self.report_data["TID连续性审计"] = stats.get("reason", "无法审计")
            return
        self.report_data["TID计数器缺口"] = (
            f"{stats.get('counter_gap_events', 0)} 处；"
            f"按可观测步长推定 {stats.get('inferred_missing_positions', 0)} 个计数位置"
        )
        ratio = stats.get("missing_ratio", float("nan"))
        missing_s = stats.get("missing_duration_s", float("nan"))
        expected_s = stats.get("expected_duration_s", float("nan"))
        qualifier = (
            "（存在非法值或不确定跳变，仅为已知部分估计）"
            if stats.get("ratio_is_partial_estimate") else ""
        )
        ratio_text = f"{ratio * 100:.6f}%" if np.isfinite(ratio) else "无法计算"
        self.report_data["TID推定缺样占完整时长"] = (
            f"{ratio_text}{qualifier}（约 {missing_s:.6f} s / {expected_s:.6f} s）"
            if np.isfinite(missing_s) and np.isfinite(expected_s)
            else f"{ratio_text}{qualifier}"
        )
        self.report_data["TID审计边界"] = (
            "TID 仅按 1..60000 循环且逐样本加一的工程假设做独立连续性审计；"
            "该语义尚未以厂家协议确认；"
            "计数缺口不等于已证实丢帧，也不能确定设备、链路或采集端原因。"
        )
        if self.source_tid_continuity.get("available") and (
                self.trim_minutes > 0 or
                self.source_tid_continuity.get("observed_valid_values") !=
                stats.get("observed_valid_values")):
            source = self.source_tid_continuity
            self.report_data["原始文件TID摘要"] = (
                f"{source.get('counter_gap_events', 0)} 处计数器缺口；"
                f"按可观测步长推定 "
                f"{source.get('inferred_missing_positions', 0)} 个计数位置；"
                f"占推定完整时长 "
                + (
                    f"{source.get('missing_ratio') * 100:.6f}%"
                    if np.isfinite(source.get('missing_ratio', float('nan')))
                    else "无法计算"
                )
            )
    
    def _apply_trim(self):
        """掐头去尾：丢弃开头和结尾各 N 分钟的数据"""
        if self.trim_minutes <= 0 or self.sample_rate <= 0:
            return
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

    def extract_metadata(self):
        """提取数据基本信息"""
        self.n_points = len(self.df)
        self.duration = self.n_points / self.sample_rate
        
        self.report_data["采样率"] = f"{self.sample_rate} Hz"
        self.report_data["采样率来源"] = (
            "用户输入/程序配置；默认为200 Hz。"
            "由于未确认原始数值时间戳的单位和历元，程序不用其自动验证采样率。"
        )
        self.report_data["总数据点数"] = f"{self.n_points} 点"
        self.report_data["采样时长"] = f"{self.duration:.2f} 秒 ({self.duration/60:.2f} 分钟)"
        self.report_data["数据文件"] = os.path.basename(self.file_path)
        self.report_data["分析时间"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if "time" in self.df.columns:
            try:
                self.report_data["时间范围"] = f"{self.df['time'].iloc[0]:.2f} ~ {self.df['time'].iloc[-1]:.2f} s"
            except:
                pass
        
        temp_cols = []
        for col in self.df.columns:
            if "temp" in col.lower() or "温度" in col:
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
        
        timestamp_columns = [
            str(col) for col in self.df.columns
            if "timestamp" in str(col).lower() or "时间戳" in str(col)
        ]
        if timestamp_columns:
            self.report_data["时间戳列"] = ", ".join(timestamp_columns)
            self.report_data["时间戳口径"] = (
                "原始保留；未知数值单位/历元时不转换为日期，"
                "也不用于自动推断采样率。"
            )
    
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
    
    def plot_time_series(self):
        """绘制时间序列图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            if "acc_z" in self.df.columns:
                self.df["acc_z_corrected"] = self.df["acc_z"] - 9.80665

            if "time" not in self.df.columns:
                self.df["time"] = np.arange(len(self.df)) / self.sample_rate
            
            fig, axes = plt.subplots(4, 2, figsize=(16, 14))
            
            color_map = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
            
            for i, axis in enumerate(["x", "y", "z"]):
                col = f"acc_{axis}"
                if col in self.df.columns:
                    data_col = col if axis != "z" else "acc_z_corrected"
                    if data_col in self.df.columns:
                        axes[i, 0].plot(self.df["time"], self.df[data_col], color=color_map[axis], linewidth=0.5, alpha=0.8)
                axes[i, 0].set_title(f"加速度计{axis.upper()}轴时间序列", fontweight="bold", fontsize=12)
                axes[i, 0].set_ylabel("加速度 (m/s²)", fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)
                
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    axes[i, 1].plot(self.df["time"], self.df[col], color=color_map[axis], linewidth=0.5, alpha=0.8)
                axes[i, 1].set_title(f"陀螺仪{axis.upper()}轴时间序列", fontweight="bold", fontsize=12)
                axes[i, 1].set_ylabel("角速度 (°/s)", fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)
            
            for axis in ["x", "y", "z"]:
                col = f"acc_{axis}"
                if col in self.df.columns:
                    data_col = col if axis != "z" else "acc_z_corrected"
                    if data_col in self.df.columns:
                        axes[3, 0].plot(self.df["time"], self.df[data_col], label=f"{axis.upper()}", color=color_map[axis], linewidth=0.5, alpha=0.8)
            
            axes[3, 0].set_title("加速度计三轴时间序列（三合一）", fontweight="bold", fontsize=12)
            axes[3, 0].legend(loc="upper right", fontsize=9)
            axes[3, 0].set_ylabel("加速度 (m/s²)", fontsize=10)
            axes[3, 0].grid(True, alpha=0.3)
            
            for axis in ["x", "y", "z"]:
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    axes[3, 1].plot(self.df["time"], self.df[col], label=f"{axis.upper()}", color=color_map[axis], linewidth=0.5, alpha=0.8)
            
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
    
    def plot_distribution(self):
        """绘制统计分布图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            if "acc_z" in self.df.columns:
                self.df["acc_z_corrected"] = self.df["acc_z"] - 9.80665
            
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
                        
                        mean_val = data.mean()
                        std_val = data.std()
                        axes_flat[i].text(0.95, 0.95, f"均值: {mean_val:.4f}\n标准差: {std_val:.4f}",
                                         transform=axes_flat[i].transAxes, ha="right", va="top", fontsize=9,
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"))
                
                axes_flat[i].set_title(sub_titles[i], fontweight="bold", fontsize=12)
                
                if i < 3:
                    axes_flat[i].set_xlabel("加速度 (m/s²)", fontsize=10)
                else:
                    axes_flat[i].set_xlabel("角速度 (°/s)", fontsize=10)
                axes_flat[i].set_ylabel("数据点数", fontsize=10)
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, "02_统计分布分析图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            self.report_data["统计分布图"] = save_path
            return save_path
        except Exception as e:
            print(f"统计分布图绘制失败: {e}")
            self.report_data["统计分布图"] = "绘制失败"
            return None
    
    def extract_random_walk_coefficient(self, tau, ad, ns=None,
                                        return_details=False):
        """按本项目工程判据在 ADEV 白噪声候选区拟合 RW。"""
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
            "selection_rule": (
                "0.1<=tau<=10 s, n_terms>=3, 至少3点, "
                "|slope+0.5|<=0.3；本项目工程判据，非标准强制门槛"
            ),
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
                result["reason"] = "斜率满足本程序白噪声区工程判据"
            else:
                result["reason"] = f"拟合斜率 {slope:.3f} 偏离 -0.5，不报告为有效 RW"

        if return_details:
            return result
        return result["value"] if result["valid"] else float("nan")

    # Flicker/pink rate-noise 的 ADEV 平台关系：
    # sigma_platform = sqrt(2 ln 2 / pi) * B。
    FLICKER_ADEV_FACTOR = float(np.sqrt(2.0 * np.log(2.0) / np.pi))
    BI_REFERENCE_TAU_MIN_S = 1.0
    BI_REFERENCE_MIN_TERMS = 20
    BI_PLATFORM_TAU_MIN_S = 1.0
    BI_PLATFORM_MIN_TERMS = 5
    BI_PLATFORM_POINTS = 3
    BI_PLATFORM_MAX_ABS_SLOPE = 0.10
    BI_PLATFORM_MAX_ABS_ADJACENT_SLOPE = 0.15
    BI_PLATFORM_MAX_LOG10_RMS = 0.03

    def extract_bias_instability(self, tau, ad, ns=None):
        """仅在连续近零斜率 ADEV 平台上提取正式 BI。"""
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        if ns is None:
            ns_arr = np.full(len(tau), np.nan, dtype=float)
        else:
            ns_arr = np.asarray(ns, dtype=float)
            if len(ns_arr) != len(tau):
                ns_arr = np.full(len(tau), np.nan, dtype=float)

        mask = (np.isfinite(tau) & np.isfinite(ad) &
                (tau >= self.BI_PLATFORM_TAU_MIN_S) & (ad > 0) &
                np.isfinite(ns_arr) & (ns_arr >= self.BI_PLATFORM_MIN_TERMS))
        result = {
            "bias_instability": float("nan"),
            "platform_adev": float("nan"),
            "platform_tau": float("nan"),
            "fit_slope": float("nan"),
            "fit_residual": float("nan"),
            "fit_max_abs_adjacent_slope": float("nan"),
            "fit_tau_min": float("nan"),
            "fit_tau_max": float("nan"),
            "fit_points": 0,
            "n_terms_min": float("nan"),
            "valid": False,
            "edge_limited": False,
            "low_edge_limited": False,
            "high_edge_limited": False,
            "platform_quality_passed": False,
            "valid_for_spec_comparison": False,
            "reason": "没有足够的连续近零斜率平台；正式BI需要有效n_terms诊断",
        }
        indices = np.flatnonzero(mask)
        if len(indices) < self.BI_PLATFORM_POINTS:
            return result

        candidates = []
        window_size = self.BI_PLATFORM_POINTS
        for eligible_start in range(0, len(indices) - window_size + 1):
            win_idx = indices[eligible_start:eligible_start + window_size]
            if not np.all(np.diff(win_idx) == 1):
                continue
            t_win = tau[win_idx]
            a_win = ad[win_idx]
            n_win = ns_arr[win_idx]
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
                score = round(abs(float(slope)) + residual, 12)
                candidates.append((score, -float(np.min(n_win)), int(win_idx[0]),
                                   win_idx, float(slope), float(intercept),
                                   residual, max_adjacent))
        if not candidates:
            result["reason"] = (
                "未找到同时满足的连续平台：|3点拟合斜率|<=0.10、"
                "max|相邻斜率|<=0.15、log10残差RMS<=0.03（本项目工程判据）"
            )
            return result

        _, _, _, win_idx, slope, intercept, residual, max_adjacent = min(candidates)
        t_win = tau[win_idx]
        n_win = ns_arr[win_idx]
        center_log_tau = float(np.mean(np.log10(t_win)))
        platform_adev = float(10 ** (intercept + slope * center_log_tau))
        platform_tau = float(10 ** center_log_tau)
        low_edge_limited = bool(win_idx[0] == indices[0])
        high_edge_limited = bool(win_idx[-1] == indices[-1])
        edge_limited = low_edge_limited or high_edge_limited
        reason = (
            f"连续平台拟合（log斜率={slope:.3f}，"
            f"max|相邻斜率|={max_adjacent:.3f}，残差={residual:.3g}）；"
            "平台门槛是本项目工程判据，非标准强制值"
        )
        if low_edge_limited:
            first_index = int(win_idx[0])
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
            "bias_instability": float(platform_adev / self.FLICKER_ADEV_FACTOR),
            "platform_adev": platform_adev,
            "platform_tau": platform_tau,
            "fit_slope": slope,
            "fit_residual": residual,
            "fit_max_abs_adjacent_slope": max_adjacent,
            "fit_tau_min": float(t_win[0]),
            "fit_tau_max": float(t_win[-1]),
            "fit_points": window_size,
            "n_terms_min": float(np.min(n_win)),
            "valid": True,
            "edge_limited": edge_limited,
            "low_edge_limited": low_edge_limited,
            "high_edge_limited": high_edge_limited,
            "platform_quality_passed": True,
            # 平台质量通过不等于测试条件与厂家规格口径已匹配。
            "valid_for_spec_comparison": False,
            "reason": reason,
        })
        return result

    def extract_minimum_adev_reference(self, tau, ad, ns=None):
        """
        提取 tau>=1 s、n_terms>=20 内的最低 ADEV 折算参考。

        该值只是“最低 ADEV 等效参考”，不是正式 BI，不能用于规格判定，
        也不宣称为严格上界。
        """
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        if ns is None:
            ns_arr = np.full(len(tau), np.nan, dtype=float)
        else:
            ns_arr = np.asarray(ns, dtype=float)
            if len(ns_arr) != len(tau):
                ns_arr = np.full(len(tau), np.nan, dtype=float)
        mask = (np.isfinite(tau) & np.isfinite(ad) & np.isfinite(ns_arr) &
                (tau >= self.BI_REFERENCE_TAU_MIN_S) & (ad > 0) &
                (ns_arr >= self.BI_REFERENCE_MIN_TERMS))
        result = {
            "adev_min_ref": float("nan"),
            "tau_ref": float("nan"),
            "n_terms": float("nan"),
            "local_slope": float("nan"),
            "left_slope": float("nan"),
            "right_slope": float("nan"),
            "b_ref": float("nan"),
            "valid": False,
            "edge_limited": False,
            "status": "N/A",
            "valid_for_spec_comparison": False,
            "reason": "tau>=1 s 且 n_terms>=20 的有效点不足",
        }
        if not np.any(mask):
            return result

        indices = np.flatnonzero(mask)
        selected = int(indices[np.argmin(ad[indices])])
        eligible_pos = int(np.where(indices == selected)[0][0])

        # 局部斜率仅使用候选集中与最低点在原始曲线上连续的相邻点，
        # 避免跨过 NaN、低支持项或乱序点拼接诊断窗口。
        local_indices = [selected]
        if eligible_pos > 0 and indices[eligible_pos - 1] == selected - 1:
            local_indices.insert(0, int(indices[eligible_pos - 1]))
        if (eligible_pos + 1 < len(indices) and
                indices[eligible_pos + 1] == selected + 1):
            local_indices.append(int(indices[eligible_pos + 1]))
        local_indices = np.asarray(local_indices, dtype=int)
        if (len(local_indices) >= 2 and
                np.all(np.diff(tau[local_indices]) > 0)):
            log_t_local = np.log10(tau[local_indices])
            log_a_local = np.log10(ad[local_indices])
            local_slope = float(np.polyfit(log_t_local, log_a_local, 1)[0])
        else:
            local_slope = float("nan")

        left_slope = float("nan")
        if (eligible_pos > 0 and indices[eligible_pos - 1] == selected - 1 and
                tau[selected] > tau[selected - 1]):
            left_slope = float(
                (np.log10(ad[selected]) - np.log10(ad[selected - 1])) /
                (np.log10(tau[selected]) - np.log10(tau[selected - 1]))
            )
        right_slope = float("nan")
        if (eligible_pos + 1 < len(indices) and
                indices[eligible_pos + 1] == selected + 1 and
                tau[selected + 1] > tau[selected]):
            right_slope = float(
                (np.log10(ad[selected + 1]) - np.log10(ad[selected])) /
                (np.log10(tau[selected + 1]) - np.log10(tau[selected]))
            )

        edge_limited = bool(eligible_pos == 0 or eligible_pos == len(indices) - 1)
        reason = "受约束最低 ADEV 折算；非正式 BI，不可用于规格判定"
        if edge_limited:
            reason += "；最低点位于筛选范围边界"
        result.update({
            "adev_min_ref": float(ad[selected]),
            "tau_ref": float(tau[selected]),
            "n_terms": float(ns_arr[selected]),
            "local_slope": local_slope,
            "left_slope": float(left_slope),
            "right_slope": float(right_slope),
            "b_ref": float(ad[selected] / self.FLICKER_ADEV_FACTOR),
            "valid": True,
            "edge_limited": edge_limited,
            "status": "仅参考（非正式BI/不可规格判定）",
            "valid_for_spec_comparison": False,
            "reason": reason,
        })
        return result

    # ----------------------------------------------------------------
    # 10 s 非重叠分段均值标准差（工程统计量）
    # ----------------------------------------------------------------
    def calculate_bias_stability_10s(self):
        """按10 s非重叠窗口计算分段均值的总体标准差（ddof=0）。"""
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
        """旧接口兼容包装；不宣称对任何 GJB 条款的合规性。"""
        return self.calculate_bias_stability_10s()

    def _gjb_10s_smoothing_std(self, data, window):
        """10 s 非重叠分段均值标准差数值计算。
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
        return float(np.std(seg_means, ddof=0))
    
    def plot_allan_variance(self):
        """计算并绘制经典非重叠 Allan 偏差（ADEV）。"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            # Allan 使用全量观测行，不截断数据。
            n_points = len(self.df)
             
            # 提取原始速率序列；不做线性去趋势。
            acc_data = {}
            gyro_data = {}
            
            for axis in ["x", "y", "z"]:
                col = f"acc_{axis}"
                # Allan 始终输入原始加速度速率序列；Z轴减重力仅供时序/分布展示。
                data_col = col
                if data_col in self.df.columns:
                    raw_data = self.df[data_col][:n_points].to_numpy(
                        dtype=np.float64, copy=False
                    )
                    raw_data = raw_data[np.isfinite(raw_data)]
                    acc_data[axis] = raw_data
                
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    raw_data = self.df[col][:n_points].to_numpy(
                        dtype=np.float64, copy=False
                    )
                    raw_data = raw_data[np.isfinite(raw_data)]
                    gyro_data[axis] = raw_data
            
            # 每个传感器轴使用独立的“曲线 + 数值说明”区域。说明文字不再叠加
            # 在曲线上；曲线绘图区固定为宽:高=6:4，六个单元保持等尺寸。
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
            
            fs_float = float(self.sample_rate)
            acc_bias = {}
            gyro_bias = {}
            acc_results = {}
            gyro_results = {}
            curve_rows = []
            summary_rows = []

            self.report_data["Allan分析配置"] = (
                f"估计器=ADEV（经典非重叠）；输入=全量观测行；"
                f"采样率={fs_float:g} Hz（用户/程序配置，未由未知单位时间戳验证）；"
                f"未线性去趋势、未插值；显示网格={ALLAN_DISPLAY_TAU_GRID}；"
                f"参数估计网格={ALLAN_ESTIMATION_TAU_GRID}。"
                "显示加密仅改变曲线尺度取样，BI/RW/最低ADEV参考仍用octave结果和原判据；"
                "n_terms≥20、5≤n_terms<20、n_terms<5的20/5分级仅为本项目工程可视化门槛，非标准规定。"
            )
            self.report_data["Allan预处理"] = (
                "原始六轴速率序列；Z轴减重力仅用于时序/分布展示，不进入Allan。"
            )
            self.report_data["Allan支持项显示门槛"] = (
                "n_terms>=20正常实线；5<=n_terms<20淡色虚线；"
                "n_terms<5灰色低支持尾部；20/5为本项目工程可视化门槛，非标准规定"
            )
            tid_stats = self.tid_continuity
            tid_audit_available = bool(tid_stats.get("available", False))
            if tid_audit_available:
                self.report_data["Allan缺样策略"] = (
                    f"TID可观测计数器缺口 {tid_stats.get('counter_gap_events', 0)} 处，"
                    f"推定 {tid_stats.get('inferred_missing_positions', 0)} 个计数位置；"
                    "计数缺口不等于已证实丢帧。Allan按观测行等间隔计算，未重建缺失时点。"
                )
                tid_gap_events = int(tid_stats.get("counter_gap_events", 0))
                tid_missing_positions = int(tid_stats.get("inferred_missing_positions", 0))
            else:
                self.report_data["Allan缺样策略"] = (
                    f"TID连续性不可审计（{tid_stats.get('reason', '原因未知')}）；"
                    "不能据此填写零缺口。Allan仍按观测行等间隔计算，未重建缺失时点。"
                )
                tid_gap_events = np.nan
                tid_missing_positions = np.nan
            tid_audit_semantics = (
                "engineering_assumption_TID_1_to_60000_increment_by_1;"
                "not_vendor_protocol_confirmed;not_proven_packet_loss"
            )

            # 10 s 非重叠分段均值标准差是独立的工程统计量。
            # 基于全量原始数据计算，供信息框多单位展示；不将其宣称为标准合规结果。
            window_10s = max(1, int(round(10 * fs_float)))
            acc_gjb_std = {}
            gyro_gjb_std = {}
            for axis in ["x", "y", "z"]:
                col = f"acc_{axis}"
                if col in self.df.columns:
                    raw = self.df[col].to_numpy(dtype=np.float64, copy=False)
                    raw = raw[np.isfinite(raw)]
                    acc_gjb_std[axis] = self._gjb_10s_smoothing_std(raw, window_10s)
                col = f"gyro_{axis}"
                if col in self.df.columns:
                    raw = self.df[col].to_numpy(dtype=np.float64, copy=False)
                    raw = raw[np.isfinite(raw)]
                    gyro_gjb_std[axis] = self._gjb_10s_smoothing_std(raw, window_10s)

            all_axes_data = []
            for i in range(6):
                ax = axes[i]
                info_ax = info_axes[i]
                
                if i < 3:  # 加速度计
                    axis = ["x", "y", "z"][i]
                    if axis not in acc_data or len(acc_data[axis]) < 100:
                        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue
                    
                    data = acc_data[axis]
                else:  # 陀螺仪
                    axis = ["x", "y", "z"][i-3]
                    if axis not in gyro_data or len(gyro_data[axis]) < 100:
                        ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue
                    
                    data = gyro_data[axis]
                
                # AllanTools 的 adev 返回 Allan 偏差而非方差。
                # 显示与参数估计使用两套独立 tau 网格：
                # - 显示/03_Allan曲线数据.csv：名义15点/十倍程 + octave锚点；
                # - BI/RW/最低ADEV参考：仍为原 octave 网格。
                try:
                    requested_display_taus = build_allan_display_taus(
                        len(data), fs_float,
                        points_per_decade=ALLAN_DISPLAY_POINTS_PER_DECADE,
                    )
                    taus, adev, adev_error, n_terms = at.adev(
                        data, rate=fs_float, data_type="freq",
                        taus=requested_display_taus,
                    )
                    taus = np.asarray(taus, dtype=float)
                    adev = np.asarray(adev, dtype=float)
                    adev_error = np.asarray(adev_error, dtype=float)
                    n_terms = np.asarray(n_terms, dtype=float)
                    valid_curve = (
                        np.isfinite(taus) & np.isfinite(adev) &
                        (taus > 0) & (adev > 0)
                    )
                    if not np.any(valid_curve):
                        ax.text(0.5, 0.5, '计算失败', ha='center', va='center',
                                transform=ax.transAxes)
                        ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue
                    # 绘图和曲线 CSV 使用高密显示网格。
                    display_taus_raw = taus
                    display_adev_raw = adev
                    display_n_terms_raw = (
                        n_terms if len(n_terms) == len(display_taus_raw)
                        else np.full(len(display_taus_raw), np.nan)
                    )
                    taus = display_taus_raw[valid_curve]
                    adev = display_adev_raw[valid_curve]
                    adev_error = (
                        adev_error[valid_curve] if len(adev_error) == len(valid_curve)
                        else np.full(len(taus), np.nan)
                    )
                    n_terms = display_n_terms_raw[valid_curve]
                    
                    if len(taus) == 0 or len(adev) == 0:
                        ax.text(0.5, 0.5, '计算失败', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                        info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                     fontsize=7, color='#666666', transform=info_ax.transAxes)
                        continue

                    # 单独重算 octave 网格供参数提取，不把显示加密点
                    # 送入原有“连续3点”平台判据或 RW 拟合。
                    estimation_taus, estimation_adev, _, estimation_n_terms = at.adev(
                        data, rate=fs_float, data_type="freq",
                        taus=ALLAN_ESTIMATION_TAU_GRID,
                    )
                    estimation_taus = np.asarray(estimation_taus, dtype=float)
                    estimation_adev = np.asarray(estimation_adev, dtype=float)
                    estimation_n_terms = np.asarray(estimation_n_terms, dtype=float)
                    if len(estimation_n_terms) != len(estimation_taus):
                        estimation_n_terms = np.full(len(estimation_taus), np.nan)
                    
                    # 提取随机游走系数（斜率拟合方法）
                    rw_result = self.extract_random_walk_coefficient(
                        estimation_taus, estimation_adev,
                        ns=estimation_n_terms, return_details=True
                    )
                    rw = rw_result["value"] if rw_result["valid"] else np.nan

                    # 正式 BI 仅由连续近零斜率平台提取，B = sigma_platform / 0.66428247。
                    bi_result = self.extract_bias_instability(
                        estimation_taus, estimation_adev, ns=estimation_n_terms
                    )
                    bi = bi_result["bias_instability"] if bi_result["valid"] else np.nan
                    min_ref_result = self.extract_minimum_adev_reference(
                        estimation_taus, estimation_adev, ns=estimation_n_terms
                    )

                    # 绘制未平滑的真实计算点。n_terms 分类只改样式，
                    # 不删点、不插值，也不修改 ADEV 数值。
                    display_support = classify_allan_support(n_terms)
                    normal_support = display_support == "normal_n_terms_ge_20"
                    limited_support = display_support == "limited_n_terms_5_to_19"
                    low_support = display_support == "low_n_terms_lt_5"
                    unknown_support = display_support == "unknown_n_terms"
                    if np.any(normal_support):
                        ax.loglog(
                            taus, np.where(normal_support, adev, np.nan),
                            linewidth=1.2, linestyle='-', color=colors[i],
                            marker=None,
                            label='ADEV（n_terms≥20）',
                        )
                    if np.any(limited_support):
                        ax.loglog(
                            taus, np.where(limited_support, adev, np.nan),
                            linewidth=1.1, linestyle='--', color=colors[i], alpha=0.48,
                            marker=None,
                            label='ADEV有限支持（5≤n_terms<20）',
                        )
                    if np.any(low_support):
                        ax.loglog(
                            taus, np.where(low_support, adev, np.nan),
                            linewidth=1.0, linestyle=':', color='#808080', alpha=0.9,
                            marker=None,
                            label='ADEV低支持尾部（n_terms<5）',
                        )
                    if np.any(unknown_support):
                        ax.loglog(
                            taus, np.where(unknown_support, adev, np.nan),
                            linewidth=1.0, linestyle=':', color='#808080', alpha=0.7,
                            marker=None,
                            label='ADEV支持项数未知',
                        )

                    # τ≈1 s 点仅用于定位，不等同于由白噪声区拟合得到的 RW 系数。
                    idx_1s = np.argmin(np.abs(taus - 1.0))
                    if taus.min() <= 1.0 <= taus.max():
                        ax.scatter(taus[idx_1s], adev[idx_1s], color='red', s=80, zorder=5,
                                  marker='o', edgecolors='darkred', linewidths=2,
                                  label=f'近1 s ADEV（实际 {taus[idx_1s]:.3g} s）')

                    if bi_result["valid"]:
                        ax.scatter(bi_result["platform_tau"], bi_result["platform_adev"],
                                   color='blue', s=80, zorder=5, marker='s',
                                   edgecolors='darkblue', linewidths=2, label='BI平台拟合')
                    if min_ref_result["valid"]:
                        ax.scatter(min_ref_result["tau_ref"], min_ref_result["adev_min_ref"],
                                   color='#e83e8c', s=90, zorder=6, marker='v',
                                   edgecolors='#8b1a50', linewidths=1.5,
                                   label='最低ADEV折算参考（非BI）')

                    # 添加参数信息文本框（中文定义，带单位转换 - 正确换算）
                    if i < 3:  # 加速度计
                        # 10 s 非重叠分段均值标准差，多单位展示。
                        gjb_std = acc_gjb_std.get(axis, float('nan'))
                        if not np.isnan(gjb_std):
                            gjb_mg = gjb_std * 1e3 / 9.80665
                            gjb_ug = gjb_std * 1e6 / 9.80665
                            gjb_text = (f"10s非重叠分段均值标准差（工程统计量）:\n"
                                        f"  {gjb_std:.6e} m/s²\n"
                                        f"  {gjb_mg:.6f} mg\n"
                                        f"  {gjb_ug:.5f} μg")
                        else:
                            gjb_text = "10s非重叠分段均值标准差（工程统计量）:\n  数据不足"

                        rw_ug_sqrt_s = rw * 1e6 / 9.80665 if np.isfinite(rw) else np.nan
                        rw_one_sided_asd_ug_hz = (
                            np.sqrt(2.0) * rw_ug_sqrt_s
                            if np.isfinite(rw_ug_sqrt_s) else np.nan
                        )
                        rw_m_s_h = rw * 60.0 if np.isfinite(rw) else np.nan
                        bi_g = bi / 9.80665 if np.isfinite(bi) else np.nan
                        bi_mg = bi * 1e3 / 9.80665 if np.isfinite(bi) else np.nan
                        bi_ug = bi * 1e6 / 9.80665 if np.isfinite(bi) else np.nan
                        b_ref = min_ref_result["b_ref"] if min_ref_result["valid"] else np.nan
                        ref_mg = b_ref * 1e3 / 9.80665 if np.isfinite(b_ref) else np.nan

                        bi_status = (
                            "边界受限（规格可比性未确认）"
                            if bi_result.get("edge_limited") else
                            ("平台识别通过（规格可比性未确认）"
                             if bi_result["valid"] else "不可提取")
                        )
                        if bi_result["valid"]:
                            bi_value_text = (
                                f"  {bi:.6e} m/s²\n"
                                f"  {bi_g:.6e} g\n"
                                f"  {bi_mg:.6f} mg\n"
                                f"  {bi_ug:.5f} μg\n"
                            )
                        else:
                            bi_value_text = "  N/A\n"
                        info_text = (f"速度随机游走 VRW:\n"
                                    f"  {rw:.6e} m/s/√s\n"
                                    f"  {rw_ug_sqrt_s:.5f} μg·√s\n"
                                    f"  {rw_m_s_h:.4e} m/s/√h\n"
                                    f"  单边白噪声ASD等效: "
                                    f"{rw_one_sided_asd_ug_hz:.5f} μg/√Hz\n"
                                    f"零偏不稳定性 BI（{bi_status}）:\n"
                                    f"{bi_value_text}"
                                    f"最低ADEV等效参考（非BI）:\n"
                                    f"  {ref_mg:.6f} mg, tau={min_ref_result['tau_ref']:.4g}s, "
                                    f"n={min_ref_result['n_terms']:.0f}\n"
                                    f"{gjb_text}")

                        # 保存结果（mg单位）
                        acc_bias[axis] = (f"{bi_mg:.6f} mg" if bi_result["valid"]
                                          else f"N/A（{bi_result['reason']}）")
                        rw_units = {"(m/s²)·√s": rw, "m/s/√s": rw,
                                    "m/s/√h": rw_m_s_h,
                                    "mg·√s": rw * 1e3 / 9.80665 if np.isfinite(rw) else np.nan,
                                    "μg·√s": rw_ug_sqrt_s,
                                    "单边ASD_(m/s²)/√Hz": (
                                        np.sqrt(2.0) * rw if np.isfinite(rw) else np.nan
                                    ),
                                    "单边ASD_mg/√Hz": (
                                        np.sqrt(2.0) * rw * 1e3 / 9.80665
                                        if np.isfinite(rw) else np.nan
                                    ),
                                    "单边ASD_μg/√Hz": rw_one_sided_asd_ug_hz}
                        bi_units = {"m/s²": bi, "g": bi_g, "mg": bi_mg, "μg": bi_ug}
                        min_ref_units = {"m/s²": b_ref,
                                         "g": b_ref / 9.80665 if np.isfinite(b_ref) else np.nan,
                                         "mg": ref_mg,
                                         "μg": b_ref * 1e6 / 9.80665 if np.isfinite(b_ref) else np.nan}
                        bs_units = {"m/s²": gjb_std,
                                    "g": gjb_std / 9.80665 if np.isfinite(gjb_std) else np.nan,
                                    "mg": gjb_mg if np.isfinite(gjb_std) else np.nan,
                                    "μg": gjb_ug if np.isfinite(gjb_std) else np.nan}
                        acc_results[axis] = {"rw": rw_result, "bi": bi_result,
                                             "min_ref": min_ref_result, "rw_units": rw_units,
                                             "bi_units": bi_units, "min_ref_units": min_ref_units,
                                             "bs_units": bs_units}
                    else:  # 陀螺仪
                        arw_dph = rw * 60.0 if np.isfinite(rw) else np.nan
                        rw_one_sided_asd_dps_hz = (
                            np.sqrt(2.0) * rw if np.isfinite(rw) else np.nan
                        )
                        bi_dph = bi * 3600.0 if np.isfinite(bi) else np.nan
                        b_ref = min_ref_result["b_ref"] if min_ref_result["valid"] else np.nan
                        ref_dph = b_ref * 3600.0 if np.isfinite(b_ref) else np.nan

                        gjb_std = gyro_gjb_std.get(axis, float('nan'))
                        if not np.isnan(gjb_std):
                            gjb_dph = gjb_std * 3600.0
                            gjb_text = (f"10s非重叠分段均值标准差（工程统计量）:\n"
                                        f"  {gjb_std:.6e} °/s\n"
                                        f"  {gjb_dph:.6f} °/h")
                        else:
                            gjb_text = "10s非重叠分段均值标准差（工程统计量）:\n  数据不足"

                        bi_status = (
                            "边界受限（规格可比性未确认）"
                            if bi_result.get("edge_limited") else
                            ("平台识别通过（规格可比性未确认）"
                             if bi_result["valid"] else "不可提取")
                        )
                        if bi_result["valid"]:
                            bi_value_text = (
                                f"  {bi:.6e} °/s\n"
                                f"  {bi_dph:.6f} °/h\n"
                            )
                        else:
                            bi_value_text = "  N/A\n"
                        info_text = (f"角度随机游走 ARW:\n"
                                    f"  {rw:.6e} °/√s\n"
                                    f"  {arw_dph:.4f} °/√h\n"
                                    f"  单边白噪声ASD等效: "
                                    f"{rw_one_sided_asd_dps_hz:.6e} (°/s)/√Hz\n"
                                    f"零偏不稳定性 BI（{bi_status}）:\n"
                                    f"{bi_value_text}"
                                    f"最低ADEV等效参考（非BI）:\n"
                                    f"  {ref_dph:.6f} °/h, tau={min_ref_result['tau_ref']:.4g}s, "
                                    f"n={min_ref_result['n_terms']:.0f}\n"
                                    f"{gjb_text}")

                        # 保存结果（°/h单位）
                        gyro_bias[axis] = (f"{bi_dph:.6f} °/h" if bi_result["valid"]
                                           else f"N/A（{bi_result['reason']}）")
                        rw_units = {"°/√s": rw, "°/√h": arw_dph,
                                    "rad/√s": rw*np.pi/180 if np.isfinite(rw) else np.nan,
                                    "rad/√h": rw*np.pi/3 if np.isfinite(rw) else np.nan,
                                    "单边ASD_(°/s)/√Hz": rw_one_sided_asd_dps_hz}
                        bi_units = {"°/s": bi, "°/h": bi_dph,
                                    "rad/s": bi*np.pi/180 if np.isfinite(bi) else np.nan,
                                    "rad/h": bi*20*np.pi if np.isfinite(bi) else np.nan}
                        min_ref_units = {"°/s": b_ref, "°/h": ref_dph,
                                         "rad/s": b_ref*np.pi/180 if np.isfinite(b_ref) else np.nan,
                                         "rad/h": b_ref*20*np.pi if np.isfinite(b_ref) else np.nan}
                        bs_units = {"°/s": gjb_std,
                                    "°/h": gjb_dph if np.isfinite(gjb_std) else np.nan,
                                    "rad/s": gjb_std*np.pi/180 if np.isfinite(gjb_std) else np.nan,
                                    "rad/h": gjb_std*20*np.pi if np.isfinite(gjb_std) else np.nan}
                        gyro_results[axis] = {"rw": rw_result, "bi": bi_result,
                                              "min_ref": min_ref_result, "rw_units": rw_units,
                                              "bi_units": bi_units, "min_ref_units": min_ref_units,
                                              "bs_units": bs_units}

                    # 数值说明放入独立轴，并在较长的说明字段上自动换行，避免
                    # 文字超出本轴宽度或遮挡曲线。数值/单位行保持原样，便于读取。
                    wrapped_info_lines = []
                    for info_line in info_text.splitlines():
                        if len(info_line) > 43:
                            wrapped_info_lines.extend(textwrap.wrap(
                                info_line, width=43, break_long_words=True,
                                break_on_hyphens=False,
                            ))
                        else:
                            wrapped_info_lines.append(info_line)
                    wrapped_info_text = "\n".join(wrapped_info_lines)
                    info_ax.text(
                        0.015, 0.985, wrapped_info_text,
                        transform=info_ax.transAxes, fontsize=6.5,
                        linespacing=1.12, ha='left', va='top', clip_on=True,
                        wrap=True,
                        color='#2f2f2f',
                    )
                    
                    ax.set_title(f'Allan Deviation - {data_names[i]}', fontweight="bold", fontsize=10)
                    ax.set_xlabel('tau (s)', fontsize=8)
                    unit = "m/s²" if i < 3 else "°/s"
                    ax.set_ylabel(f'Allan Deviation ({unit})', fontsize=8)
                    ax.tick_params(axis='both', which='both', labelsize=7)
                    ax.grid(True, which='both', alpha=0.3)
                    ax.legend(fontsize=6.4, loc='best', framealpha=0.82,
                              borderpad=0.35, handlelength=2.4, labelspacing=0.3)
                    
                    all_axes_data.append((taus, adev, rw, bi))
                    sensor = "acc" if i < 3 else "gyro"
                    estimation_m = set(
                        np.rint(estimation_taus * fs_float).astype(np.int64).tolist()
                    )
                    display_m = np.rint(taus * fs_float).astype(np.int64)
                    for j in range(len(taus)):
                        is_parameter_tau = bool(display_m[j] in estimation_m)
                        curve_rows.append({
                            "sensor": sensor, "axis": axis,
                            "tau_s": float(taus[j]), "adev": float(adev[j]),
                            "error": float(adev_error[j]) if np.isfinite(adev_error[j]) else np.nan,
                            "n_terms": float(n_terms[j]) if np.isfinite(n_terms[j]) else np.nan,
                            "grid_role": "display_curve",
                            "tau_grid_role": (
                                "display_and_parameter_octave_anchor"
                                if is_parameter_tau else "display_only"
                            ),
                            "is_parameter_tau": is_parameter_tau,
                            "display_tau_grid": ALLAN_DISPLAY_TAU_GRID,
                            "display_points_per_decade": ALLAN_DISPLAY_POINTS_PER_DECADE,
                            "support_class": str(display_support[j]),
                            "display_support_class": str(display_support[j]),
                            "estimation_tau_grid": ALLAN_ESTIMATION_TAU_GRID,
                            "parameter_estimation_source": "separate_octave_grid",
                            "method": "ADEV（经典非重叠）", "sample_rate_hz": fs_float,
                            "sample_rate_source": "user_config_not_timestamp_verified",
                            "preprocessing": "raw_no_linear_detrend",
                            "gap_policy": "observed_rows_assumed_regular",
                            "allan_error_semantics": "AllanTools_approximate_error_not_confidence_interval",
                            "tid_audit_available": tid_audit_available,
                            "tid_audit_scope": "current_analysis_rows_after_numeric_cleaning_and_trim",
                            "tid_audit_reason": tid_stats.get("reason", ""),
                            "tid_audit_semantics": tid_audit_semantics,
                            "tid_gap_events": tid_gap_events,
                            "tid_inferred_missing_positions": tid_missing_positions,
                        })
                    summary_rows.append({
                        "sensor": sensor, "axis": axis,
                        "input_unit": "m/s²" if sensor == "acc" else "°/s",
                        "method": "ADEV（经典非重叠）", "sample_rate_hz": fs_float,
                        "sample_rate_source": "user_config_not_timestamp_verified",
                        "preprocessing": "raw_no_linear_detrend",
                        "gap_policy": "observed_rows_assumed_regular",
                        "allan_error_semantics": "AllanTools_approximate_error_not_confidence_interval",
                        "tid_audit_available": tid_audit_available,
                        "tid_audit_scope": "current_analysis_rows_after_numeric_cleaning_and_trim",
                        "tid_audit_reason": tid_stats.get("reason", ""),
                        "tid_audit_semantics": tid_audit_semantics,
                        "display_tau_grid": ALLAN_DISPLAY_TAU_GRID,
                        "display_points_per_decade": ALLAN_DISPLAY_POINTS_PER_DECADE,
                        "display_tau_min_s": float(np.min(taus)),
                        "display_tau_max_s": float(np.max(taus)),
                        "display_curve_points": int(len(taus)),
                        "display_normal_support_points": int(np.count_nonzero(normal_support)),
                        "display_limited_support_points": int(np.count_nonzero(limited_support)),
                        "display_low_support_points": int(np.count_nonzero(low_support)),
                        "estimation_tau_grid": ALLAN_ESTIMATION_TAU_GRID,
                        "parameter_estimation_source": "separate_octave_grid",
                        "parameter_tau_min_s": float(np.min(estimation_taus)),
                        "parameter_tau_max_s": float(np.max(estimation_taus)),
                        "parameter_curve_points": int(len(estimation_taus)),
                        "tau_min_s": float(np.min(taus)), "tau_max_s": float(np.max(taus)),
                        "curve_points": int(len(taus)),
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
                        "bi_platform_detected": bool(bi_result["valid"]),
                        "bi_value_input_unit": float(bi) if np.isfinite(bi) else np.nan,
                        "bi_platform_adev": float(bi_result.get("platform_adev", np.nan)),
                        "bi_platform_tau_s": float(bi_result.get("platform_tau", np.nan)),
                        "bi_fit_slope": float(bi_result.get("fit_slope", np.nan)),
                        "bi_fit_residual_log10_rms": float(bi_result.get("fit_residual", np.nan)),
                        "bi_fit_max_abs_adjacent_slope": float(
                            bi_result.get("fit_max_abs_adjacent_slope", np.nan)
                        ),
                        "bi_fit_tau_min_s": float(bi_result.get("fit_tau_min", np.nan)),
                        "bi_fit_tau_max_s": float(bi_result.get("fit_tau_max", np.nan)),
                        "bi_fit_points": int(bi_result.get("fit_points", 0)),
                        "bi_fit_min_n_terms": float(bi_result.get("n_terms_min", np.nan)),
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
                        "min_ref_available": bool(min_ref_result["valid"]),
                        "min_ref_adev": float(min_ref_result.get("adev_min_ref", np.nan)),
                        "min_ref_equivalent_bias": float(min_ref_result.get("b_ref", np.nan)),
                        "min_ref_tau_s": float(min_ref_result.get("tau_ref", np.nan)),
                        "min_ref_n_terms": float(min_ref_result.get("n_terms", np.nan)),
                        "min_ref_local_slope": float(min_ref_result.get("local_slope", np.nan)),
                        "min_ref_edge_limited": bool(min_ref_result.get("edge_limited", False)),
                        "min_ref_valid_for_spec_comparison": False,
                        "min_ref_tau_constraint_s": float(self.BI_REFERENCE_TAU_MIN_S),
                        "min_ref_min_terms_constraint": int(self.BI_REFERENCE_MIN_TERMS),
                        "min_ref_reason": min_ref_result.get("reason", ""),
                        "bs_10s_value_input_unit": float(gjb_std) if np.isfinite(gjb_std) else np.nan,
                        "tid_gap_events": tid_gap_events,
                        "tid_inferred_missing_positions": tid_missing_positions,
                    })
                    
                except Exception as e:
                    print(f"{data_names[i]} Allan偏差计算失败: {e}")
                    ax.text(0.5, 0.5, f'计算失败\n{str(e)}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=9)
                    ax.set_title(f'{data_names[i]}', fontweight="bold", fontsize=10)
                    info_ax.text(0.5, 0.5, '无可用数值说明', ha='center', va='center',
                                 fontsize=7, color='#666666', transform=info_ax.transAxes)
            
            # 底部添加单位换算表 (使用中文字体链，确保所有字符正确渲染)
            # 注意: 高位 Unicode 上标 (³ ⁶ ⁻) 在 Microsoft YaHei 中不存在，会显示豆腐块
            # 所以使用科学计数法 e3/e6/e-3/e-6 代替，更安全也更标准
            conv_text = (
                "═══════════════════════════════════════════════════════════════════════════════════════════\n"
                "                    Allan偏差结果与常用单位换算参考\n"
                "───────────────────────────────────────────────────────────────────────────────────\n"
                "[加速度计]  ADEV 原始单位 = m/s²\n"
                "  VRW系数N: m/s/√s → m/s/√h = ×60；→ μg·√s = ×1e6/9.80665\n"
                "  BI  (Bias Instability): B = sigma_platform / 0.66428247\n"
                "    m/s²  →  g   = 数值 / 9.80665\n"
                "    m/s²  →  mg  = 数值 × 1e3 / 9.80665\n"
                "    m/s²  →  μg  = 数值 × 1e6 / 9.80665\n"
                "───────────────────────────────────────────────────────────────────────────────────\n"
                "[陀螺仪]    ADEV 原始单位 = °/s\n"
                "  ARW系数N: °/√s → °/√h = ×60；→ rad/√s = ×π/180；rad/√h = ×π/3\n"
                "  BI  (Bias Instability): B = sigma_platform / 0.66428247\n"
                "    °/s  →  °/h = 数值 × 3600\n"
                "───────────────────────────────────────────────────────────────────────────────────\n"
                "[10s工程统计量] 数据按10s非重叠分段，取各段均值的总体标准差（ddof=0）\n"
                "  加速度计:  m/s²  →  mg  = 数值 × 1e3 / 9.80665\n"
                "  加速度计:  m/s²  →  μg  = 数值 × 1e6 / 9.80665\n"
                "  陀螺仪:    °/s   →  °/h = 数值 × 3600\n"
                "───────────────────────────────────────────────────────────────────────────────────\n"
                "条件等效单边白噪声ASD = √2×N；仅限白速率噪声及0..fs/2单边PSD约定；\n"
                "它不是直接谱估计，也不自动等同厂家噪声密度。\n"
                "曲线显示为名义15点/十倍程并加入octave锚点，不做后处理平滑；\n"
                "n_terms>=20正常实线，5..19淡色虚线，<5灰色低支持尾部。20/5是本项目工程可视化门槛，非标准规定。\n"
                "BI、RW及最低ADEV等效参考均仍使用octave网格和原有判据。\n"
                "正式BI只从连续近零斜率平台提取；无可信平台时为N/A。平台识别门槛是本项目工程判据。\n"
                "最低ADEV等效参考限定 tau>=1s、n_terms>=20；不是正式BI，不用于规格判定。\n"
                "1 g = 9.80665 m/s²    1 mg = 9.80665e-3 m/s²    1 μg = 9.80665e-6 m/s²\n"
                "═══════════════════════════════════════════════════════════════════════════════════════════"
            )
            fig.text(0.5, 0.012, conv_text, ha='center', va='bottom',
                    fontsize=6.0,
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5dc', edgecolor='#888888', alpha=0.95))

            save_path = os.path.join(self.save_dir, "03_零偏稳定性分析图.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
            plt.close()
            
            curve_path = os.path.join(self.save_dir, "03_Allan曲线数据.csv")
            summary_path = os.path.join(self.save_dir, "03_Allan参数汇总.csv")
            if curve_rows:
                pd.DataFrame(curve_rows).to_csv(curve_path, index=False, encoding="utf-8-sig")
            else:
                curve_path = "无有效Allan曲线数据"
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
            else:
                summary_path = "无有效Allan参数汇总"
            self.report_data["加速度计零偏稳定性"] = acc_bias
            self.report_data["陀螺仪零偏稳定性"] = gyro_bias
            self.report_data["加速度计零偏不稳定性_BI"] = acc_results
            self.report_data["陀螺仪零偏不稳定性_BI"] = gyro_results
            self.report_data["Allan曲线数据"] = curve_path
            self.report_data["Allan参数汇总"] = summary_path
            self.report_data["Allan方差图"] = save_path
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
    
    def plot_psd(self):
        """绘制PSD图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            G0 = 9.80665
            
            def calc_psd(data, fs, nperseg=1024):
                data = np.asarray(data)
                data = data[np.isfinite(data)]
                if len(data) < 10:
                    return None, None
                nperseg = int(min(nperseg, len(data)))
                noverlap = int(nperseg // 2)
                fs_float = float(fs) if fs is not None else 200.0
                f, psd = welch(data, fs=fs_float, nperseg=nperseg, noverlap=noverlap,
                              window="hann", detrend="constant", scaling="density", return_onesided=True)
                return f, psd
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            color_map = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
            acc_asd_factor = 1e6 / G0
            gyro_asd_factor = 1.0
            
            f_ref = None
            
            for sensor_type, row_idx in [("acc", 0), ("gyro", 1)]:
                for axis in ["x", "y", "z"]:
                    col = f"{sensor_type}_{axis}"
                    if col in self.df.columns:
                        data = self.df[col].dropna()
                        if len(data) > 100:
                            f, psd = calc_psd(data, self.sample_rate)
                            if f is not None:
                                if f_ref is None:
                                    f_ref = f
                                
                                asd_factor = acc_asd_factor if sensor_type == "acc" else gyro_asd_factor
                                asd = np.sqrt(psd) * asd_factor
                                
                                axes[row_idx, 0].semilogy(f, asd, color=color_map[axis], label=f"{axis.upper()}轴", linewidth=1)
                                axes[row_idx, 1].loglog(f[f>0], asd[f>0], color=color_map[axis], label=f"{axis.upper()}轴", linewidth=1)
            
            y_labels = {
                "acc": "ASD (µg/√Hz)",
                "gyro": "ASD (°/s/√Hz)"
            }
            titles = {
                "acc": "加速度计",
                "gyro": "陀螺仪"
            }
            
            for sensor_type, row_idx in [("acc", 0), ("gyro", 1)]:
                axes[row_idx, 0].set_title(f"{titles[sensor_type]}ASD（Y轴对数）", fontweight="bold", fontsize=13)
                axes[row_idx, 0].set_ylabel(y_labels[sensor_type], fontsize=11)
                axes[row_idx, 0].legend(fontsize=9)
                axes[row_idx, 0].grid(True, which="both", alpha=0.3)
                
                axes[row_idx, 1].set_title(f"{titles[sensor_type]}ASD（双对数）", fontweight="bold", fontsize=13)
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
            self.report_data["PSD图"] = "绘制失败"
            return None
    
    def plot_correlation(self):
        """绘制相关性分析图"""
        try:
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            acc_cols = [c for c in ["acc_x", "acc_y", "acc_z"] if c in self.df.columns]
            gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in self.df.columns]
            
            if len(acc_cols) < 2 and len(gyro_cols) < 2:
                self.report_data["相关性图"] = "数据不足"
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            if len(acc_cols) >= 2:
                acc_clean = pd.DataFrame(detrend(self.df[acc_cols].dropna(), axis=0), columns=acc_cols)
                acc_corr = acc_clean.corr()
                
                sns.heatmap(acc_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                           fmt=".4f", linewidths=0.5, square=True, annot_kws={"size": 14, "weight": "bold"}, ax=axes[0])
                axes[0].set_title("加速度计XYZ轴相关性分析", fontweight="bold", fontsize=14)
            
            if len(gyro_cols) >= 2:
                gyro_clean = pd.DataFrame(detrend(self.df[gyro_cols].dropna(), axis=0), columns=gyro_cols)
                gyro_corr = gyro_clean.corr()
                
                sns.heatmap(gyro_corr, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, center=0,
                           fmt=".4f", linewidths=0.5, square=True, annot_kws={"size": 14, "weight": "bold"}, ax=axes[1])
                axes[1].set_title("陀螺仪XYZ轴相关性分析", fontweight="bold", fontsize=14)
            
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
    
    def plot_drift(self):
        """绘制长期零漂趋势图（如果有四元数数据）"""
        try:
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

            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            
            G0 = 9.80665
            FS = self.sample_rate
            FS_INT = int(round(FS))  # 确保是整数
            DT = 1.0 / FS
            WINDOW_SIZES = [5, 10, 30, 60, 120]
            STEP_SIZE = FS_INT
            
            acc_cols = ["acc_x", "acc_y", "acc_z"]
            if not all(col in self.df.columns for col in acc_cols):
                self.report_data["漂移分析图"] = "缺少加速度计数据"
                return None
            
            acc_raw = self.df[acc_cols].values
            quats = self.df[quat_cols].values
            
            results = {s: [] for s in WINDOW_SIZES}
            
            min_window = min(WINDOW_SIZES) * FS_INT
            if len(self.df) < min_window + 100:
                self.report_data["漂移分析图"] = "数据量不足"
                return None
            
            for sec in WINDOW_SIZES:
                window_samples = sec * FS_INT
                if len(self.df) < window_samples + 100:
                    continue
                
                for start in range(0, len(self.df) - window_samples, STEP_SIZE):
                    end = start + window_samples
                    if end > len(self.df):
                        continue
                    
                    try:
                        r = R.from_quat(quats[start:end])
                        acc_w = r.apply(acc_raw[start:end])
                        acc_xy = acc_w[:, :2]
                        acc_xy_centered = acc_xy - np.mean(acc_xy, axis=0)
                        
                        vel_x = np.cumsum(acc_xy_centered[:, 0]) * DT
                        vel_y = np.cumsum(acc_xy_centered[:, 1]) * DT
                        
                        pos_x = np.cumsum(vel_x) * DT
                        pos_y = np.cumsum(vel_y) * DT
                        
                        results[sec].append([pos_x[-1], pos_y[-1]])
                    except:
                        continue
            
            self._drift_results_cache = {
                "window_sizes": tuple(WINDOW_SIZES),
                "results": results
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
            
            for i, sec in enumerate(WINDOW_SIZES):
                data = np.array(results[sec])
                if len(data) == 0:
                    continue
                
                ax1.scatter(data[:, 0], data[:, 1], s=5, alpha=0.4, color=colors[i], label=f"{sec}s数据")
                
                dist = np.linalg.norm(data, axis=1)
                if len(dist) > 0:
                    sigma_1 = np.percentile(dist, 68)
                    
                    circle = plt.Circle((0, 0), sigma_1, color=colors[i], fill=False, linestyle="--",
                                       linewidth=2, label=f"{sec}s 1σ: {sigma_1:.4f}m")
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
                # 与06_长期零漂趋势图保持同一口径：终点水平位移半径的68百分位。
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
            bars = ax.bar(seconds, radii, width=4.0, color=colors, edgecolor="#444444", linewidth=1.2)
            
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
    
    def _allan_report_cells(self, sensor: str, axis: str) -> dict:
        """把结构化 Allan 结果格式化为 HTML/Markdown 可复用的单元格。"""
        result_key = (
            "加速度计零偏不稳定性_BI"
            if sensor == "acc" else "陀螺仪零偏不稳定性_BI"
        )
        entry = self.report_data.get(result_key, {}).get(axis, {})
        if not entry:
            old_bs_key = (
                "加速度计零偏稳定性_10s平滑"
                if sensor == "acc" else "陀螺仪零偏稳定性_10s平滑"
            )
            return {
                "rw": "N/A",
                "bi": "N/A",
                "min_ref": "N/A",
                "bs": self.report_data.get(old_bs_key, {}).get(axis, "N/A"),
                "method": "未记录",
            }

        def fmt(value, unit):
            try:
                if value is None or not np.isfinite(value):
                    return "N/A"
                return f"{value:.6e} {unit}"
            except (TypeError, ValueError):
                return "N/A"

        rw_result = entry.get("rw") or {}
        bi_result = entry.get("bi") or {}
        min_ref_result = entry.get("min_ref") or {}
        rw_units = entry.get("rw_units") or {}
        bi_units = entry.get("bi_units") or {}
        min_ref_units = entry.get("min_ref_units") or {}
        bs_units = entry.get("bs_units") or {}

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
            edge_limited = bool(bi_result.get("edge_limited"))
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
                f"最小n_terms={bi_result.get('fit_min_n_terms', bi_result.get('n_terms_min', np.nan)):.0f}"
            )
            spec_note = "<br>规格可比性未确认；需另行匹配厂家测试条件与定义"
            bi_cell = f"{bi_status}<br>{bi}<br>{bi_diag}{spec_note}"
        else:
            bi_cell = f"N/A<br>{bi_result.get('reason', '未找到可信平台')}"

        if min_ref_result.get("available", min_ref_result.get("valid", False)):
            slope = min_ref_result.get("local_slope", np.nan)
            slope_text = f"{slope:.3f}" if np.isfinite(slope) else "N/A"
            edge_text = "；候选范围边界" if min_ref_result.get("edge_limited") else ""
            reason = min_ref_result.get("reason")
            reason_text = f"<br>{reason}" if reason else ""
            min_ref_cell = (
                f"参考值，非正式BI<br>{min_ref}<br>"
                f"tau={min_ref_result.get('tau', min_ref_result.get('tau_ref', np.nan)):.6g} s；"
                f"n_terms={min_ref_result.get('n_terms', np.nan):.0f}；"
                f"局部斜率={slope_text}{edge_text}{reason_text}<br>"
                "不用于规格判定"
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

    def generate_html_report(self):
        """生成HTML报告"""
        try:
            html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMU数据分析报告 — 武汉元生 yis321S</title>
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
        <h1>IMU 数据分析报告 — 武汉元生 yis321S</h1>
        <p class="subtitle">Powered by IMU Analysis Toolkit | 武汉元生 yis321S</p>
"""
            
            html += "\n        <div class='info-box'>\n            <div class='info-title'>数据基本信息</div>\n            <div class='info-grid'>\n"
            
            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位",
                "Allan曲线数据", "Allan参数汇总", "Allan结果",
                "时间序列图", "统计分布图", "Allan方差图", "Allan偏差图",
                "PSD图", "相关性图", "漂移分析图", "落点半径对比图", "落点半径对比数据",
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
                html += ("            <p><b>曲线支持项显示：</b>n_terms≥20 为正常实线，"
                         "5≤n_terms&lt;20 为淡色虚线，n_terms&lt;5 为灰色低支持尾部。"
                         "20/5 仅为本项目工程可视化门槛，非标准规定。</p>\n")
                html += "            <table class='stats-table'>\n"
                html += "                <tr><th>指标</th><th>传感器</th><th>方法/口径</th><th>单位与数值（X）</th><th>单位与数值（Y）</th><th>单位与数值（Z）</th></tr>\n"
                for sensor, label in [("gyro", "陀螺仪"), ("acc", "加速度计")]:
                    cells = [self._allan_report_cells(sensor, axis) for axis in ["x", "y", "z"]]
                    html += f"                <tr><td><b>正式零偏不稳定性 BI</b></td><td>{label}</td><td>{cells[0]['method']}<br>B=σ平台/0.66428247；连续近零斜率平台<br>识别门槛为本项目工程判据</td>"
                    html += "".join(f"<td>{cell['bi']}</td>" for cell in cells) + "</tr>\n"
                    html += f"                <tr><td><b>最低 ADEV 等效参考</b><br><span style='color:#b83280;'>非正式 BI</span></td><td>{label}</td><td>tau≥1 s、n_terms≥20；σmin/0.66428247<br>不用于规格判定</td>"
                    html += "".join(f"<td>{cell['min_ref']}</td>" for cell in cells) + "</tr>\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    html += f"                <tr><td><b>{rw_label}</b></td><td>{label}</td><td>{cells[0]['method']}<br>白噪声区斜率拟合并折算至1 s</td>"
                    html += "".join(f"<td>{cell['rw']}</td>" for cell in cells) + "</tr>\n"
                    html += f"                <tr><td><b>10 s 非重叠分段均值标准差</b></td><td>{label}</td><td>工程统计量；非重叠分段；ddof=0</td>"
                    html += "".join(f"<td>{cell['bs']}</td>" for cell in cells) + "</tr>\n"
                html += "            </table>\n        </div>\n"

                html += """
        <div class='info-box'>
            <div class='info-title'>指标定义说明</div>
            <table class='stats-table' style='font-size:13px;'>
                <tr><th style='background:#4a5568;'>指标名称</th><th style='background:#4a5568;'>定义</th><th style='background:#4a5568;'>计算方法</th><th style='background:#4a5568;'>标准依据</th></tr>
                <tr><td><b>零偏不稳定性 (BI)</b></td>
                    <td>Allan 偏差中的 flicker/pink rate-noise 平台系数</td>
                    <td>识别连续近零斜率平台，B = σ平台 / 0.66428247；平台识别通过不等于规格可比，所有平台均需另行匹配厂家测试条件与定义，边界平台另有边界限制</td>
                    <td>噪声模型关系与工程判据；不作笼统标准合规声明</td></tr>
                <tr><td><b>最低 ADEV 等效参考</b></td>
                    <td>可信平台不明显时仍可报告的受约束工程参考；不是正式 BI</td>
                    <td>仅在 tau≥1 s 且 n_terms≥20 中取最低 ADEV，再除以 0.66428247；同时报告 tau、n_terms、局部斜率和边界状态</td>
                    <td>本项目工程判据，非标准强制门槛；不用于规格判定</td></tr>
                <tr><td><b>10 s 非重叠分段均值标准差</b></td>
                    <td>10 秒非重叠窗口均值的总体标准差</td>
                    <td>按10 s分段，计算各段均值的 std(ddof=0)</td>
                    <td>工程统计量，条款适用性需另行核对</td></tr>
                <tr><td><b>角度随机游走 (ARW)</b></td>
                    <td>陀螺仪白噪声引起的角度积分随机游走</td>
                    <td>ADEV 在白噪声候选区进行 log-log 斜率拟合，外推至 1 s</td>
                    <td>Allan 噪声模型参考</td></tr>
                <tr><td><b>速度随机游走 (VRW)</b></td>
                    <td>加速度计白噪声引起的速度积分随机游走</td>
                    <td>ADEV 在白噪声候选区进行 log-log 斜率拟合，外推至 1 s</td>
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
                ("漂移分析图", "6. 长期零漂趋势图"),
                ("落点半径对比图", "7. 不同时间窗口1σ落点半径对比图")
            ]
            
            for key, title in image_keys:
                if key in self.report_data:
                    img_path = self.report_data[key]
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        html += f"\n        <div class='image-section'>\n            <div class='image-title'>{title}</div>\n            <img src='{img_name}' class='result-image' alt='{title}' />\n        </div>\n"
                    elif isinstance(img_path, str):
                        html += f"\n        <div class='image-section'>\n            <div class='image-title'>{title}</div>\n            <div class='warning'>{img_path}</div>\n        </div>\n"

            html += """
        <div class='image-section'>
            <div class='image-title'>Allan 偏差常用单位换算参考表</div>
            <table class='stats-table' style='font-size: 13px;'>
                <tr><th style='background:#4a5568;'>参数</th><th style='background:#4a5568;'>计算单位</th><th style='background:#4a5568;'>换算公式</th><th style='background:#4a5568;'>目标单位</th></tr>
                <tr style='background:#fef9e7;'><td rowspan='3'><b>加速度计 VRW 系数 N</b><br><span style='color:#666;'>(Velocity Random Walk)</span></td>
                    <td rowspan='3'>m/s/√s（即 (m/s²)·√s）</td>
                    <td>= N</td><td>m/s/√s</td></tr>
                <tr style='background:#fef9e7;'><td>= N × 60</td><td>m/s/√h</td></tr>
                <tr style='background:#fef9e7;'><td>= N × 1×10⁶ / 9.80665</td><td>μg·√s</td></tr>
                <tr style='background:#fff8dc;'><td rowspan='3'><b>加速度计 BI/等效参考/BS</b></td>
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
                <tr style='background:#fff8dc;'><td rowspan='3'><b>陀螺仪 BI/等效参考/BS</b></td>
                    <td rowspan='3'>°/s</td>
                    <td>= 数值 × 3600</td><td>°/h</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × π / 180</td><td>rad/s</td></tr>
                <tr style='background:#fff8dc;'><td>= 数值 × 20π</td><td>rad/h</td></tr>
            </table>
            <div style='margin-top:10px; padding:10px; background:#f0f4f8; border-left:4px solid #667eea; font-size:13px;'>
                <b>说明：</b><br>
                1. <b>输入同量纲</b>：ADEV 的单位由输入速率序列决定，图轴不因结果换算而改变。<br>
                2. <b>BI</b>：仅对识别出的连续近零斜率平台计算，B = σ平台 / 0.66428247；无可信平台时为 N/A。<br>
                3. <b>最低 ADEV 等效参考</b>：只在 tau≥1 s、n_terms≥20 中取最低点；不是正式 BI，不用于规格判定。<br>
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
            md = "# IMU 数据分析报告 — 武汉元生 yis321S\n\n"
            md += "---\n\n## 数据基本信息\n\n"
            
            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位",
                "Allan曲线数据", "Allan参数汇总", "Allan结果",
                "时间序列图", "统计分布图", "Allan方差图", "Allan偏差图",
                "PSD图", "相关性图", "漂移分析图", "落点半径对比图", "落点半径对比数据",
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
                md += ("曲线支持项显示：n_terms≥20 为正常实线，5≤n_terms<20 为淡色虚线，"
                       "n_terms<5 为灰色低支持尾部。20/5 仅为本项目工程可视化门槛，非标准规定。\n\n")
                md += "| 指标 | 传感器 | 方法/口径 | X轴 | Y轴 | Z轴 |\n"
                md += "|------|------|------|-----|-----|-----|\n"
                for sensor, label in [("gyro", "陀螺仪"), ("acc", "加速度计")]:
                    cells = [self._allan_report_cells(sensor, axis) for axis in ["x", "y", "z"]]
                    md += f"| **正式零偏不稳定性 BI** | {label} | {cells[0]['method']}；B=σ平台/0.66428247；连续近零斜率平台；识别门槛为本项目工程判据 |"
                    for cell in cells:
                        md += f" {cell['bi']} |"
                    md += "\n"
                    md += f"| **最低 ADEV 等效参考（非正式 BI）** | {label} | tau≥1 s、n_terms≥20；σmin/0.66428247；不用于规格判定 |"
                    for cell in cells:
                        md += f" {cell['min_ref']} |"
                    md += "\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    md += f"| **{rw_label}** | {label} | {cells[0]['method']}；白噪声区斜率拟合并折算至1 s |"
                    for cell in cells:
                        md += f" {cell['rw']} |"
                    md += "\n"
                    md += f"| **10 s 非重叠分段均值标准差** | {label} | 工程统计量；非重叠分段；ddof=0 |"
                    for cell in cells:
                        md += f" {cell['bs']} |"
                    md += "\n"

                md += """
### 指标定义说明

| 指标名称 | 定义 | 计算方法 | 标准依据 |
|----------|------|----------|----------|
| **零偏不稳定性 (BI)** | Allan 偏差中的 flicker/pink rate-noise 平台系数 | 识别连续近零斜率平台，B=σ平台/0.66428247；平台识别通过不等于规格可比，所有平台均需匹配厂家测试条件，边界平台另有边界限制 | 噪声模型关系与本项目工程判据；不作笼统标准合规声明 |
| **最低 ADEV 等效参考** | 无可信平台时仍可报告的受约束工程参考；不是正式 BI | tau≥1 s且n_terms≥20的最低ADEV / 0.66428247；报告tau、n_terms、局部斜率和边界状态 | 本项目工程判据，非标准强制门槛；不用于规格判定 |
| **10 s 非重叠分段均值标准差** | 10 秒非重叠窗口均值的总体标准差 | 分段均值后计算 std(ddof=0) | 工程统计量，条款适用性需另行核对 |
| **角度随机游走 (ARW)** | 陀螺仪白噪声引起的角度积分随机游走 | ADEV 在白噪声候选区拟合斜率并外推至1 s | Allan 噪声模型参考 |
| **速度随机游走 (VRW)** | 加速度计白噪声引起的速度积分随机游走 | ADEV 在白噪声候选区拟合斜率并外推至1 s | Allan 噪声模型参考 |
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
                ("漂移分析图", "6. 长期零漂趋势图"),
                ("落点半径对比图", "7. 不同时间窗口1σ落点半径对比图")
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

**说明：**

1. **输入同量纲**：ADEV 的单位由输入速率序列决定，图轴保持原坐标单位。
2. **BI**：仅对连续近零斜率平台计算，B=σ平台/0.66428247；无可信平台时为 N/A。
3. **最低 ADEV 等效参考**：仅在 tau≥1 s、n_terms≥20 的候选中选取；它不是正式 BI，不用于规格判定。
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
    
    def run_full_analysis(self, progress_callback=None):
        """运行完整分析"""
        steps = [
            ("计算基本统计", self.calculate_basic_stats),
            ("计算10秒分段均值标准差", self.calculate_bias_stability_10s),
            ("绘制时间序列图", self.plot_time_series),
            ("绘制统计分布图", self.plot_distribution),
            ("绘制Allan偏差图", self.plot_allan_variance),
            ("绘制PSD图", self.plot_psd),
            ("绘制相关性分析图", self.plot_correlation),
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


class IMUAnalysisApp:
    """主HMI界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("IMU 数据分析工具箱 — 武汉元生 yis321S")
        self.root.geometry("800x600")
        
        self.file_path = tk.StringVar()
        self.sample_rate = tk.DoubleVar(value=200.0)
        self.trim_minutes = tk.StringVar(value="0")
        self.analyzing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="IMU 数据分析工具箱 — 武汉元生 yis321S",
                                font=("Microsoft YaHei", 20, "bold"))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame,
                                  text="武汉元生 yis321S 一站式静态数据分析解决方案",
                                  font=("Microsoft YaHei", 10),
                                  foreground="#718096")
        subtitle_label.pack(pady=(0, 25))
        
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(file_frame, text="数据文件:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.file_path, width=60, font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=10)
        ttk.Button(file_frame, text="浏览...", command=self.browse_file).pack(side=tk.LEFT)
        
        sr_frame = ttk.Frame(main_frame)
        sr_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(sr_frame, text="采样率 (Hz):", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        ttk.Entry(sr_frame, textvariable=self.sample_rate, width=15, font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=10)
        ttk.Label(sr_frame, text="（默认200Hz）", font=("Microsoft YaHei", 9), foreground="#718096").pack(side=tk.LEFT)

        # 掐头去尾输入行
        trim_frame = ttk.Frame(main_frame)
        trim_frame.pack(fill=tk.X, pady=8)
        ttk.Label(trim_frame, text="掐头去尾 (分钟):", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        ttk.Entry(trim_frame, textvariable=self.trim_minutes, width=6, font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=10)
        ttk.Label(trim_frame, text="（输入 0 或不填则不丢弃；例如输入 5，丢弃头尾各 5 分钟）",
                 font=("Microsoft YaHei", 9), foreground="#718096").pack(side=tk.LEFT)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.analyze_btn = ttk.Button(btn_frame, text="开始完整分析", command=self.start_analysis, width=25)
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
            "✓ Allan偏差分析（平台BI + 受约束最低ADEV参考）",
            "✓ 功率谱密度（PSD）分析",
            "✓ 相关性分析（热图）",
            "✓ 长期零漂趋势分析",
            "✓ 不同时间窗口1σ落点半径对比",
            "✓ 生成HTML和Markdown报告",
            "✓ 增强鲁棒性：支持各种数据格式差异"
        ]
        
        for feature in features:
            ttk.Label(info_frame, text=feature, font=("Microsoft YaHei", 10)).pack(anchor="w", pady=3)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="选择IMU数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
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
            sr_val = self.sample_rate.get()
            if sr_val <= 0:
                raise ValueError("采样率必须大于0")
        except Exception as e:
            messagebox.showerror("错误", f"采样率输入错误: {e}\n请输入一个有效的数字（如200）")
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


if __name__ == "__main__":
    root = tk.Tk()
    app = IMUAnalysisApp(root)
    root.mainloop()
