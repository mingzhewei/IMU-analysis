# 整合的IMU分析工具箱 — 无锡凌思 LINS16460 专用版
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


def build_allan_display_taus(n_samples, sample_rate, points_per_decade=15):
    """公开的 Allan 显示 tau 网格入口，便于独立回归测试。"""
    return IMUDataAnalyzer._build_allan_display_tau_grid(
        n_samples, sample_rate, points_per_decade
    )


def classify_allan_support(n_terms):
    """按支持项数量分类；分类只影响显示样式，不删改曲线点。"""
    ns = np.asarray(n_terms, dtype=float)
    support = np.full(ns.shape, "low_n_terms_lt_5", dtype=object)
    support[np.isfinite(ns) & (ns >= 20)] = "normal_n_terms_ge_20"
    support[np.isfinite(ns) & (ns >= 5) & (ns < 20)] = (
        "limited_n_terms_5_to_19"
    )
    return support

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


class IMUDataAnalyzer:
    """IMU数据分析核心类"""
    
    def __init__(self, file_path, sample_rate=200, trim_minutes=0, allan_method="adev"):
        self.file_path = file_path
        self.sample_rate = float(sample_rate)  # 确保采样率是浮点数
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("采样率必须是大于 0 的有限数值")
        self.trim_minutes = float(trim_minutes)  # 掐头去尾分钟数，0 表示不截断
        # 默认使用经典非重叠 ADEV；调用方可显式选择 OADEV。
        # 两种估计器都会在报告和导出的 CSV 中记录，避免混用计算口径。
        method = str(allan_method).lower()
        self.allan_method = method if method in {"adev", "oadev"} else "adev"
        self.save_dir = os.path.join(os.path.dirname(file_path), "analysis_results")
        self.report_data = {}
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.load_data()
    
    def load_data(self):
        """加载数据并提取基本信息 - LINS16450/16460 原始采集格式（Tab 分隔，5 行表头）"""
        encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]

        # LINS16450/16460 原始采集文件格式：
        #   第1行 设备信息 / 第2行 采集时间 / 第3行 空行 / 第4行 列名 / 第5行 单位
        #   数据为 Tab 分隔，共 10 列：
        #   Ax Ay Az (g) | Gx Gy Gz (dps) | Roll Pitch Yaw (degrees) | TempX (deg_C)
        col_names_16450 = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz",
                           "Roll", "Pitch", "Yaw", "TempX"]

        self.df = None
        read_exception = None
        for enc in encodings:
            try:
                # 注意：该格式每行末尾带有多余的 Tab（会产生一个空尾列），
                # 不能在此传 names（列数不匹配时 pandas 会把首列误当索引，导致整体错位），
                # 先按无列名读入，再截取前 10 列赋标准列名
                self.df = pd.read_csv(self.file_path, encoding=enc, sep="\t",
                                     skipinitialspace=True, skiprows=5, header=None,
                                     low_memory=False, on_bad_lines="error")
                print(f"成功使用编码 {enc} 加载数据")
                break
            except Exception as e:
                read_exception = e
                self.df = None
                print(f"使用编码 {enc} 失败: {e}")
                continue

        if self.df is None:
            raise ValueError(
                "无法完整加载 LINS16460 数据文件；程序不会静默跳过字段数异常的行。"
                f"请检查编码、5 行表头和 Tab 列结构。最后错误: {read_exception}"
            )

        self.report_data["数据坏行策略"] = (
            "字段数异常严格报错；六轴非数值或非有限行单独计数后剔除，不插值"
        )
        source_row_count = int(len(self.df))

        if self.df.shape[1] < len(col_names_16450):
            raise ValueError(
                f"数据列不足：需要至少 {len(col_names_16450)} 个有效列，"
                f"实际读取到 {self.df.shape[1]} 列"
            )
        if self.df.shape[1] > len(col_names_16450):
            extra_columns = self.df.iloc[:, len(col_names_16450):]
            if bool(extra_columns.notna().to_numpy().any()):
                raise ValueError(
                    "检测到第10个有效列之后仍有非空数据；该文件不符合"
                    "LINS16450/16460 的10列结构，未继续截断或错位解析"
                )

        # 截取前 10 列（丢弃行尾多余 Tab 产生的空列）并赋标准列名
        self.df = self.df.iloc[:, :len(col_names_16450)]
        self.df.columns = col_names_16450

        self.df = self.df.loc[:, ~self.df.columns.astype(str).str.match(r"^Unnamed")]
        self.df.columns = self.df.columns.astype(str).str.strip()
        self.df = self.df.loc[:, self.df.columns != ""]

        # 列名映射：LINS16450/16460 原始列名 → 分析标准列名
        col_mapping = {
            "Ax": "acc_x",
            "Ay": "acc_y",
            "Az": "acc_z",
            "Gx": "gyro_x",
            "Gy": "gyro_y",
            "Gz": "gyro_z",
            "Roll": "roll",
            "Pitch": "pitch",
            "Yaw": "yaw",
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

        if missing_cols:
            raise ValueError(
                f"缺少六轴 Allan 分析必需列: {missing_cols}\n"
                f"需要的列: {required_cols}\n找到的列: {list(self.df.columns)}"
            )

        for col in available_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            except:
                pass

        # 欧拉角列转数值（长期零漂分析使用）
        for col in ["roll", "pitch", "yaw"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # 单位换算：加速度计 g → m/s²（LINS16450/16460 输出单位为 g，
        # 后续全部分析统一在 m/s² 下进行；陀螺仪 dps 即 °/s，无需换算）
        G0 = 9.80665
        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in self.df.columns:
                self.df[col] = self.df[col] * G0

        six_axis_values = self.df[required_cols].to_numpy(dtype=float, copy=False)
        invalid_numeric_mask = pd.Series(
            ~np.isfinite(six_axis_values).all(axis=1), index=self.df.index
        )
        invalid_numeric_rows = int(invalid_numeric_mask.sum())
        self.report_data["源数据行数"] = source_row_count
        self.report_data["六轴非数值或非有限行数"] = invalid_numeric_rows
        self.report_data["数值坏行处理"] = (
            "无" if invalid_numeric_rows == 0 else
            f"剔除 {invalid_numeric_rows} 行六轴非数值记录；未插值，后续按保留观测行计算"
        )
        self.df = self.df.loc[~invalid_numeric_mask].reset_index(drop=True)
        
        if len(self.df) < 100:
            raise ValueError(f"有效数据点太少！仅 {len(self.df)} 个点")
        
        self._apply_trim()
        self.extract_metadata()
    
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
            "user_config_not_file_verified；LINS16460 文件无逐样本时间戳/帧号，"
            "采样率由用户输入或程序配置，不能由该文件独立验证"
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
        
        for col in self.df.columns:
            if "timestamp" in col.lower() or "时间戳" in col:
                try:
                    t = pd.to_datetime(self.df[col], errors="coerce").dropna()
                    if len(t) > 0:
                        self.report_data["开始时间"] = t.min().strftime("%Y-%m-%d %H:%M:%S")
                        self.report_data["结束时间"] = t.max().strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
    
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
    
    # ----------------------------------------------------------------
    # Allan 偏差（ADEV/OADEV）参数提取
    # ----------------------------------------------------------------
    # 显示曲线使用较密的对数 tau 网格；参数提取保持独立 octave 网格，
    # 以免增加绘图点数时无意改变 BI/RW 的计算口径。
    ALLAN_DISPLAY_POINTS_PER_DECADE = 15
    ALLAN_DISPLAY_GRID_NAME = "nominal_15_points_per_decade_plus_octave_anchors"
    ALLAN_ESTIMATION_GRID_NAME = "octave"
    ALLAN_DISPLAY_NORMAL_MIN_TERMS = 20
    ALLAN_DISPLAY_LOW_MIN_TERMS = 5

    @staticmethod
    def _build_allan_display_tau_grid(sample_count, sample_rate,
                                      points_per_decade=15):
        """生成显示用 tau，并在整数平均因子上并入全部 octave 锚点。"""
        n = int(sample_count)
        rate = float(sample_rate)
        density = int(points_per_decade)
        if n < 2 or not np.isfinite(rate) or rate <= 0 or density <= 0:
            return np.array([], dtype=float)
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
        """兼容外部回归测试的显示 tau 网格别名。"""
        return cls._build_allan_display_tau_grid(
            sample_count, sample_rate, points_per_decade
        )

    @classmethod
    def _allan_support_masks(cls, n_terms):
        ns = np.asarray(n_terms, dtype=float)
        normal = np.isfinite(ns) & (ns >= cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS)
        caution = (np.isfinite(ns) & (ns >= cls.ALLAN_DISPLAY_LOW_MIN_TERMS) &
                   (ns < cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS))
        low = ~(normal | caution)
        return normal, caution, low

    @classmethod
    def _allan_support_class(cls, n_terms):
        if np.isfinite(n_terms) and n_terms >= cls.ALLAN_DISPLAY_NORMAL_MIN_TERMS:
            return "normal_n_terms_ge_20"
        if np.isfinite(n_terms) and n_terms >= cls.ALLAN_DISPLAY_LOW_MIN_TERMS:
            return "limited_n_terms_5_to_19"
        return "low_n_terms_lt_5"

    def extract_random_walk_coefficient(self, tau, ad, ns=None,
                                        return_details: bool = False):
        """在 0.1～10 s 白噪声候选区拟合随机游走系数。

        ADEV 白速率噪声区的 log-log 斜率应接近 -1/2。拟合曲线在
        tau=1 s 的值为随机游走系数；不满足本程序工程斜率门槛时只保留
        诊断，不把结果标记为有效。门槛不是标准强制值。
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
            "value": float("nan"), "slope": float("nan"),
            "intercept": float("nan"), "tau_min": float("nan"),
            "tau_max": float("nan"), "n_points": int(np.count_nonzero(mask)),
            "valid": False, "reason": "白噪声拟合区有效点不足（需要至少3点）",
        }
        if result["n_points"] >= 3:
            slope, intercept = np.polyfit(np.log10(tau[mask]), np.log10(ad[mask]), 1)
            result.update({
                "value": float(10 ** intercept), "slope": float(slope),
                "intercept": float(intercept), "tau_min": float(np.min(tau[mask])),
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
        """返回受约束最低 ADEV 及其等效偏置参考；该值不是正式 BI。"""
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        ns_arr = np.asarray(ns, dtype=float) if ns is not None else np.full(len(tau), np.nan)
        if len(ns_arr) != len(tau):
            ns_arr = np.full(len(tau), np.nan)
        mask = (np.isfinite(tau) & np.isfinite(ad) & (ad > 0) &
                (tau >= self.BI_REFERENCE_TAU_MIN_S) & np.isfinite(ns_arr) &
                (ns_arr >= self.BI_REFERENCE_MIN_TERMS))
        result = {
            "available": False, "adev": float("nan"), "tau": float("nan"),
            "n_terms": float("nan"), "equivalent_bias": float("nan"),
            "local_slope": float("nan"), "edge_limited": False,
            "valid_for_spec_comparison": False,
            "tau_min_constraint_s": float(self.BI_REFERENCE_TAU_MIN_S),
            "min_terms_constraint": int(self.BI_REFERENCE_MIN_TERMS),
            "reason": (f"未找到同时满足 tau>={self.BI_REFERENCE_TAU_MIN_S:g} s 且 "
                       f"n_terms>={self.BI_REFERENCE_MIN_TERMS} 的有限 ADEV 点"),
        }
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return result
        selected = int(indices[np.argmin(ad[indices])])
        eligible_pos = int(np.where(indices == selected)[0][0])
        local_indices = [selected]
        if eligible_pos > 0 and indices[eligible_pos - 1] == selected - 1:
            local_indices.insert(0, int(indices[eligible_pos - 1]))
        if eligible_pos + 1 < len(indices) and indices[eligible_pos + 1] == selected + 1:
            local_indices.append(int(indices[eligible_pos + 1]))
        local_indices = np.asarray(local_indices, dtype=int)
        local_slope = float("nan")
        if (len(local_indices) >= 2 and np.all(np.diff(tau[local_indices]) > 0) and
                np.all(ad[local_indices] > 0)):
            local_slope = float(np.polyfit(
                np.log10(tau[local_indices]), np.log10(ad[local_indices]), 1)[0])
        edge_limited = bool(eligible_pos == 0 or eligible_pos == len(indices) - 1)
        reason = (f"受约束最低ADEV（tau>={self.BI_REFERENCE_TAU_MIN_S:g} s，"
                  f"n_terms>={self.BI_REFERENCE_MIN_TERMS}）；仅作等效参考，不是平台BI")
        if edge_limited:
            reason += "；最低点位于候选范围边界"
        result.update({
            "available": True, "curve_index": selected, "adev": float(ad[selected]),
            "tau": float(tau[selected]), "n_terms": float(ns_arr[selected]),
            "equivalent_bias": float(ad[selected] / self.FLICKER_ADEV_FACTOR),
            "local_slope": local_slope, "edge_limited": edge_limited,
            "reason": reason,
        })
        return result

    def extract_bias_instability(self, tau, ad, ns=None):
        """识别连续近零斜率平台，并按 B=sigma_platform/0.66428 提取 BI。"""
        tau = np.asarray(tau, dtype=float)
        ad = np.asarray(ad, dtype=float)
        ns_arr = np.asarray(ns, dtype=float) if ns is not None else np.full(len(tau), np.nan)
        if len(ns_arr) != len(tau):
            ns_arr = np.full(len(tau), np.nan)
        mask = (np.isfinite(tau) & np.isfinite(ad) & (ad > 0) &
                (tau >= self.BI_PLATFORM_TAU_MIN_S) & np.isfinite(ns_arr) &
                (ns_arr >= self.BI_PLATFORM_MIN_TERMS))
        result = {
            "bias_instability": float("nan"), "min_adev": float("nan"),
            "tau_at_min": float("nan"), "platform_adev": float("nan"),
            "platform_tau": float("nan"), "fit_slope": float("nan"),
            "fit_residual": float("nan"), "fit_max_abs_adjacent_slope": float("nan"),
            "fit_tau_min": float("nan"), "fit_tau_max": float("nan"),
            "fit_points": 0, "fit_min_n_terms": float("nan"), "valid": False,
            "edge_limited": False, "low_edge_limited": False,
            "high_edge_limited": False, "platform_quality_passed": False,
            "valid_for_spec_comparison": False,
            "reason": "没有足够的连续近零斜率平台；正式BI需要有效n_terms诊断",
        }
        indices = np.flatnonzero(mask)
        if len(indices) < self.BI_PLATFORM_POINTS:
            return result
        candidates = []
        for start in range(0, len(indices) - self.BI_PLATFORM_POINTS + 1):
            win = indices[start:start + self.BI_PLATFORM_POINTS]
            if not np.all(np.diff(win) == 1):
                continue
            t_win, a_win, n_win = tau[win], ad[win], ns_arr[win]
            if not np.all(np.diff(t_win) > 0):
                continue
            log_t, log_a = np.log10(t_win), np.log10(a_win)
            adjacent = np.diff(log_a) / np.diff(log_t)
            slope, intercept = np.polyfit(log_t, log_a, 1)
            residual = float(np.sqrt(np.mean((log_a - (slope * log_t + intercept)) ** 2)))
            max_adjacent = float(np.max(np.abs(adjacent)))
            if (abs(slope) <= self.BI_PLATFORM_MAX_ABS_SLOPE and
                    max_adjacent <= self.BI_PLATFORM_MAX_ABS_ADJACENT_SLOPE and
                    residual <= self.BI_PLATFORM_MAX_LOG10_RMS):
                score = round(abs(float(slope)) + residual, 12)
                candidates.append((score, -float(np.min(n_win)), int(win[0]), win,
                                   float(slope), float(intercept), residual, max_adjacent))
        if not candidates:
            result["reason"] = (
                "未找到同时满足的连续平台：abs(3点拟合斜率)<=0.10、"
                "max|相邻斜率|<=0.15、log10残差RMS<=0.03（本项目工程判据）"
            )
            return result
        _, _, _, selected, slope, intercept, residual, max_adjacent = min(candidates)
        t_win, n_win = tau[selected], ns_arr[selected]
        center_log_tau = float(np.mean(np.log10(t_win)))
        platform_adev = float(10 ** (intercept + slope * center_log_tau))
        platform_tau = float(10 ** center_log_tau)
        low_edge = bool(selected[0] == indices[0])
        high_edge = bool(selected[-1] == indices[-1])
        reason = (f"连续平台拟合（log斜率={slope:.3f}，"
                  f"max|相邻斜率|={max_adjacent:.3f}，残差={residual:.3g}）；"
                  "平台门槛是本项目工程判据，非标准强制值")
        if low_edge:
            reason += "；平台位于有效候选范围左边界，需复核短 tau 噪声区"
        if high_edge:
            reason += "；平台受记录长度/支持项上界限制，需更长数据确认"
        result.update({
            "bias_instability": float(platform_adev / self.FLICKER_ADEV_FACTOR),
            "min_adev": platform_adev, "tau_at_min": platform_tau,
            "platform_adev": platform_adev, "platform_tau": platform_tau,
            "fit_slope": slope, "fit_residual": residual,
            "fit_max_abs_adjacent_slope": max_adjacent,
            "fit_tau_min": float(t_win[0]), "fit_tau_max": float(t_win[-1]),
            "fit_points": int(self.BI_PLATFORM_POINTS),
            "fit_min_n_terms": float(np.min(n_win)), "valid": True,
            "edge_limited": bool(low_edge or high_edge),
            "low_edge_limited": low_edge, "high_edge_limited": high_edge,
            "platform_quality_passed": True, "reason": reason,
        })
        return result

    # ----------------------------------------------------------------
    # 10 s 非重叠分段均值标准差（工程统计量）
    # ----------------------------------------------------------------
    def calculate_bias_stability_10s(self):
        """按 10 s 非重叠窗口计算分段均值的总体标准差（ddof=0）。"""
        G0 = 9.80665
        window = max(1, int(round(10 * self.sample_rate)))
        acc_display, gyro_display, acc_raw, gyro_raw = {}, {}, {}, {}
        for axis in ["x", "y", "z"]:
            for sensor, raw_store, display_store in (
                    ("acc", acc_raw, acc_display), ("gyro", gyro_raw, gyro_display)):
                data = self.df[f"{sensor}_{axis}"].to_numpy(dtype=float, copy=False)
                data = data[np.isfinite(data)]
                if len(data) < 2 * window:
                    continue
                n_seg = len(data) // window
                means = np.array([np.mean(data[i * window:(i + 1) * window])
                                  for i in range(n_seg)])
                value = float(np.std(means, ddof=0))
                raw_store[axis] = value
                display_store[axis] = (f"{value * 1e3 / G0:.6f} mg" if sensor == "acc"
                                       else f"{value * 3600.0:.6f} °/h")
        self.report_data["加速度计零偏稳定性_10s平滑"] = acc_display
        self.report_data["陀螺仪零偏稳定性_10s平滑"] = gyro_display
        self.report_data["10s分段统计口径"] = (
            "全量保留观测行；10 s非重叠分段均值的总体标准差(ddof=0)；工程统计量"
        )
        self.report_data["加速度计BS_10s单位"] = {
            axis: {"m/s²": v, "g": v / G0, "mg": v * 1e3 / G0,
                   "μg": v * 1e6 / G0} for axis, v in acc_raw.items()
        }
        self.report_data["陀螺仪BS_10s单位"] = {
            axis: {"°/s": v, "°/h": v * 3600.0, "rad/s": v * np.pi / 180.0,
                   "rad/h": v * 20.0 * np.pi} for axis, v in gyro_raw.items()
        }
        return acc_display, gyro_display

    def calculate_bias_stability_gjb(self):
        """旧接口兼容包装；实际执行 10 s 分段均值工程统计。"""
        return self.calculate_bias_stability_10s()

    def _segment_mean_std_10s(self, data, window):
        """10 s 非重叠分段均值标准差；数据不足两个窗口时返回 NaN。"""
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if window <= 0 or len(data) < 2 * window:
            return float("nan")
        n_seg = len(data) // window
        means = np.array([np.mean(data[k * window:(k + 1) * window])
                          for k in range(n_seg)])
        return float(np.std(means, ddof=0))

    def _gjb_10s_smoothing_std(self, data, window):
        """旧私有接口兼容包装；不表示或声称 GJB 合规。"""
        return self._segment_mean_std_10s(data, window)

    def plot_allan_variance(self):
        """用全量原始速率序列绘制 ADEV/OADEV，并导出可追溯参数。"""
        try:
            for key in (
                "Allan曲线数据", "Allan参数汇总", "Allan方差图", "Allan偏差图",
                "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
            ):
                self.report_data.pop(key, None)
            plt.close("all")
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = list(_CN_FALLBACK_NAMES) + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            G0 = 9.80665
            fs = float(self.sample_rate)
            method = str(getattr(self, "allan_method", "adev")).lower()
            method = method if method in {"adev", "oadev"} else "adev"
            estimator = at.oadev if method == "oadev" else at.adev
            method_label = "OADEV（重叠）" if method == "oadev" else "ADEV（经典非重叠）"
            self.report_data["Allan分析配置"] = (
                f"估计器={method_label}；输入=全量保留观测行；采样率={fs:g} Hz；"
                "未线性去趋势、未插值；显示网格=每十倍程约15点并保留全部octave锚点；"
                "BI/RW/最低ADEV等效参考的参数提取网格=octave；"
                "n_terms的20/5分级仅为本项目工程可视化门槛，非标准规定"
            )
            self.report_data["Allan曲线支持项显示"] = (
                "n_terms≥20正常显示；5≤n_terms<20提醒显示；n_terms<5低支持尾部；"
                "20/5为本项目工程可视化门槛，非标准规定"
            )
            self.report_data["Allan显示网格"] = (
                "nominal 15 points/decade plus octave anchors；仅增加显示和曲线CSV细节，"
                "未进行曲线平滑"
            )
            self.report_data["Allan参数提取网格"] = (
                "octave；BI、RW及最低ADEV等效参考均只使用该独立网格"
            )
            self.report_data["Allan预处理"] = (
                "原始速率序列（加速度 m/s²、陀螺 °/s）；未减均值、未线性去趋势"
            )
            self.report_data["Allan时基限制"] = (
                "源文件无逐样本时间戳/帧号；按用户配置采样率和保留观测行等间隔计算；"
                "无法由文件判断缺帧，未重建缺失时间点"
            )

            fig, axes = plt.subplots(2, 3, figsize=(22, 12))
            axes = axes.flatten()
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            names = ["acc_x(m/s²)", "acc_y(m/s²)", "acc_z(m/s²)",
                     "gyro_x(°/s)", "gyro_y(°/s)", "gyro_z(°/s)"]
            units = ["m/s²"] * 3 + ["°/s"] * 3
            curve_rows, summary_rows = [], []
            acc_results, gyro_results, acc_bias, gyro_bias = {}, {}, {}, {}
            window_10s = max(1, int(round(10 * fs)))
            acc_bs = {a: self._segment_mean_std_10s(
                self.df[f"acc_{a}"].to_numpy(dtype=float, copy=False), window_10s)
                for a in ["x", "y", "z"]}
            gyro_bs = {a: self._segment_mean_std_10s(
                self.df[f"gyro_{a}"].to_numpy(dtype=float, copy=False), window_10s)
                for a in ["x", "y", "z"]}

            def fmt(v, unit="", digits=6):
                return "N/A" if not np.isfinite(v) else f"{v:.{digits}e} {unit}".rstrip()

            def acc_units(v):
                return {"m/s²": v, "g": v / G0 if np.isfinite(v) else np.nan,
                        "mg": v * 1e3 / G0 if np.isfinite(v) else np.nan,
                        "μg": v * 1e6 / G0 if np.isfinite(v) else np.nan}

            def gyro_units(v):
                return {"°/s": v, "°/h": v * 3600 if np.isfinite(v) else np.nan,
                        "rad/s": v * np.pi / 180 if np.isfinite(v) else np.nan,
                        "rad/h": v * 20 * np.pi if np.isfinite(v) else np.nan}

            def acc_rw_units(v):
                asd = v * np.sqrt(2) if np.isfinite(v) else np.nan
                return {"(m/s²)·√s": v, "m/s/√s": v,
                        "m/s/√h": v * 60 if np.isfinite(v) else np.nan,
                        "mg·√s": v * 1e3 / G0 if np.isfinite(v) else np.nan,
                        "μg·√s": v * 1e6 / G0 if np.isfinite(v) else np.nan,
                        "单边ASD_(m/s²)/√Hz": asd,
                        "单边ASD_μg/√Hz": asd * 1e6 / G0 if np.isfinite(asd) else np.nan}

            def gyro_rw_units(v):
                asd = v * np.sqrt(2) if np.isfinite(v) else np.nan
                return {"°/√s": v, "°/√h": v * 60 if np.isfinite(v) else np.nan,
                        "rad/√s": v * np.pi / 180 if np.isfinite(v) else np.nan,
                        "rad/√h": v * np.pi / 3 if np.isfinite(v) else np.nan,
                        "单边ASD_(°/s)/√Hz": asd}

            for i, ax in enumerate(axes):
                sensor = "acc" if i < 3 else "gyro"
                axis = ["x", "y", "z"][i if i < 3 else i - 3]
                col = f"{sensor}_{axis}"
                ax.set_title(names[i], fontweight="bold", fontsize=12)
                try:
                    raw = self.df[col].to_numpy(dtype=np.float64, copy=False)
                    finite = np.isfinite(raw)
                    if np.count_nonzero(finite) < 100:
                        ax.text(0.5, 0.5, "数据不足", ha="center", va="center",
                                transform=ax.transAxes)
                        continue
                    data = raw[finite]
                    display_tau_request = self._build_allan_display_tau_grid(
                        len(data), fs, self.ALLAN_DISPLAY_POINTS_PER_DECADE
                    )
                    display_taus, display_adev, display_errors, display_n_terms = estimator(
                        data, rate=fs, data_type="freq", taus=display_tau_request)
                    estimation_taus, estimation_adev, estimation_errors, estimation_n_terms = estimator(
                        data, rate=fs, data_type="freq", taus=self.ALLAN_ESTIMATION_GRID_NAME)
                    display_taus = np.asarray(display_taus, float)
                    display_adev = np.asarray(display_adev, float)
                    display_errors = np.asarray(display_errors, float)
                    display_n_terms = np.asarray(display_n_terms, float)
                    estimation_taus = np.asarray(estimation_taus, float)
                    estimation_adev = np.asarray(estimation_adev, float)
                    estimation_n_terms = np.asarray(estimation_n_terms, float)
                    valid = (np.isfinite(display_taus) & np.isfinite(display_adev) &
                             (display_taus > 0) & (display_adev > 0))
                    if not np.any(valid):
                        ax.text(0.5, 0.5, "计算失败", ha="center", va="center",
                                transform=ax.transAxes)
                        continue
                    taus_v, adev_v = display_taus[valid], display_adev[valid]
                    err_v = (display_errors[valid] if len(display_errors) == len(display_taus)
                             else np.full(len(taus_v), np.nan))
                    ns_v = (display_n_terms[valid] if len(display_n_terms) == len(display_taus)
                            else np.full(len(taus_v), np.nan))
                    ns_extract = (estimation_n_terms
                                  if len(estimation_n_terms) == len(estimation_taus)
                                  else np.full(len(estimation_taus), np.nan))
                    rw_result = self.extract_random_walk_coefficient(
                        estimation_taus, estimation_adev, ns_extract, return_details=True)
                    bi_result = self.extract_bias_instability(
                        estimation_taus, estimation_adev, ns_extract)
                    min_ref_result = self.extract_minimum_adev_reference(
                        estimation_taus, estimation_adev, ns_extract)
                    rw = rw_result["value"] if rw_result["valid"] else np.nan
                    bi = bi_result["bias_instability"] if bi_result["valid"] else np.nan
                    min_ref = (min_ref_result["equivalent_bias"]
                               if min_ref_result["available"] else np.nan)

                    normal, caution, low = self._allan_support_masks(ns_v)
                    ax.loglog(taus_v, adev_v, linewidth=1.6, color=colors[i], alpha=0.45)
                    if np.any(normal):
                        ax.loglog(taus_v[normal], adev_v[normal], linewidth=2,
                                  color=colors[i], label=f"{method_label}（n_terms≥20）")
                    if np.any(caution):
                        ax.scatter(taus_v[caution], adev_v[caution], s=20,
                                   facecolors="none", edgecolors="#f59e0b", marker="o",
                                   label="支持项5～19")
                    if np.any(low):
                        ax.scatter(taus_v[low], adev_v[low], s=24, color="#9ca3af",
                                   marker="x", label="支持项<5")
                    idx_1s = int(np.argmin(np.abs(taus_v - 1.0)))
                    if taus_v.min() <= 1 <= taus_v.max():
                        ax.scatter(taus_v[idx_1s], adev_v[idx_1s], color="red", s=70,
                                   marker="o", label=f"近1 s（实际{taus_v[idx_1s]:.3g}s）")
                    if bi_result["valid"]:
                        ax.scatter(bi_result["platform_tau"], bi_result["platform_adev"],
                                   color="blue", s=75, marker="s", label="BI平台拟合")
                    if min_ref_result["available"]:
                        ax.scatter(min_ref_result["tau"], min_ref_result["adev"],
                                   color="#e83e8c", s=80, marker="v",
                                   label="最低ADEV折算参考（非BI）")

                    bs_value = (acc_bs if sensor == "acc" else gyro_bs)[axis]
                    bi_u = acc_units(bi) if sensor == "acc" else gyro_units(bi)
                    min_u = acc_units(min_ref) if sensor == "acc" else gyro_units(min_ref)
                    rw_u = acc_rw_units(rw) if sensor == "acc" else gyro_rw_units(rw)
                    bs_u = acc_units(bs_value) if sensor == "acc" else gyro_units(bs_value)
                    bi_text = ("N/A（未识别到可信平台）" if not bi_result["valid"] else
                               (fmt(bi_u["mg"], "mg") if sensor == "acc"
                                else fmt(bi_u["°/h"], "°/h")))
                    min_text = ("N/A" if not min_ref_result["available"] else
                                (fmt(min_u["mg"], "mg") if sensor == "acc"
                                 else fmt(min_u["°/h"], "°/h")))
                    if sensor == "acc":
                        rw_text = fmt(rw_u["m/s/√h"], "m/s/√h")
                        bs_text = fmt(bs_u["mg"], "mg")
                    else:
                        rw_text = fmt(rw_u["°/√h"], "°/√h")
                        bs_text = fmt(bs_u["°/h"], "°/h")
                    info = (f"RW: {rw_text}\n斜率: "
                            f"{rw_result['slope']:.3f} ({'有效' if rw_result['valid'] else '无效'})\n"
                            f"BI(平台/0.66428): {bi_text}\n"
                            f"最低ADEV等效参考(非BI): {min_text}\n"
                            f"BS(10s分段均值std): {bs_text}")
                    ax.text(0.04, 0.04, info, transform=ax.transAxes, fontsize=7.5,
                            verticalalignment="bottom",
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
                    ax.set_xlabel("tau (s)")
                    ax.set_ylabel(f"Allan Deviation ({units[i]})")
                    ax.grid(True, which="both", alpha=0.3)
                    ax.legend(fontsize=8, loc="upper right")

                    entry = {"rw": rw_result, "bi": bi_result, "min_ref": min_ref_result,
                             "rw_units": rw_u, "bi_units": bi_u, "min_ref_units": min_u,
                             "bs_units": bs_u}
                    (acc_results if sensor == "acc" else gyro_results)[axis] = entry
                    legacy = acc_bias if sensor == "acc" else gyro_bias
                    legacy[axis] = bi_text
                    for j in range(len(taus_v)):
                        curve_rows.append({
                            "sensor": sensor, "axis": axis, "tau_s": float(taus_v[j]),
                            "adev": float(adev_v[j]),
                            "error": float(err_v[j]) if np.isfinite(err_v[j]) else np.nan,
                            "n_terms": float(ns_v[j]) if np.isfinite(ns_v[j]) else np.nan,
                            "method": method_label, "sample_rate_hz": fs,
                            "sample_rate_source": "user_config_not_file_verified",
                            "preprocessing": "raw_no_linear_detrend",
                            "gap_policy": "observed_rows_no_reconstruction",
                            "allan_error_semantics": "AllanTools_approximate_error_not_confidence_interval",
                            "display_points_per_decade": self.ALLAN_DISPLAY_POINTS_PER_DECADE,
                            "display_tau_grid": self.ALLAN_DISPLAY_GRID_NAME,
                            "estimation_tau_grid": self.ALLAN_ESTIMATION_GRID_NAME,
                            "parameter_estimation_source": "separate_octave_grid",
                            "support_class": self._allan_support_class(ns_v[j]),
                        })
                    summary_rows.append({
                        "sensor": sensor, "axis": axis, "input_unit": units[i],
                        "method": method_label, "sample_rate_hz": fs,
                        "sample_rate_source": "user_config_not_file_verified",
                        "preprocessing": "raw_no_linear_detrend",
                        "gap_policy": "observed_rows_no_reconstruction",
                        "display_points_per_decade": self.ALLAN_DISPLAY_POINTS_PER_DECADE,
                        "display_tau_grid": self.ALLAN_DISPLAY_GRID_NAME,
                        "estimation_tau_grid": self.ALLAN_ESTIMATION_GRID_NAME,
                        "parameter_estimation_source": "separate_octave_grid",
                        "tau_min_s": float(np.min(taus_v)), "tau_max_s": float(np.max(taus_v)),
                        "curve_points": int(len(taus_v)), "rw_valid": bool(rw_result["valid"]),
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
                        "rw_slope": float(rw_result["slope"]), "rw_reason": rw_result["reason"],
                        "bi_available": bool(bi_result["valid"]),
                        "bi_value_input_unit": float(bi) if np.isfinite(bi) else np.nan,
                        "bi_platform_adev": float(bi_result["platform_adev"]),
                        "bi_platform_tau_s": float(bi_result["platform_tau"]),
                        "bi_fit_slope": float(bi_result["fit_slope"]),
                        "bi_fit_residual_log10_rms": float(bi_result["fit_residual"]),
                        "bi_fit_max_abs_adjacent_slope": float(
                            bi_result["fit_max_abs_adjacent_slope"]
                        ),
                        "bi_fit_tau_min_s": float(bi_result["fit_tau_min"]),
                        "bi_fit_tau_max_s": float(bi_result["fit_tau_max"]),
                        "bi_fit_points": int(bi_result["fit_points"]),
                        "bi_fit_min_n_terms": float(bi_result["fit_min_n_terms"]),
                        "bi_edge_limited": bool(bi_result["edge_limited"]),
                        "bi_low_edge_limited": bool(bi_result["low_edge_limited"]),
                        "bi_high_edge_limited": bool(bi_result["high_edge_limited"]),
                        "bi_platform_quality_passed": bool(
                            bi_result["platform_quality_passed"]
                        ),
                        "bi_valid_for_spec_comparison": False, "bi_reason": bi_result["reason"],
                        "min_ref_available": bool(min_ref_result["available"]),
                        "min_ref_adev": float(min_ref_result["adev"]),
                        "min_ref_equivalent_bias": float(min_ref),
                        "min_ref_tau_s": float(min_ref_result["tau"]),
                        "min_ref_n_terms": float(min_ref_result["n_terms"]),
                        "min_ref_local_slope": float(min_ref_result["local_slope"]),
                        "min_ref_edge_limited": bool(min_ref_result["edge_limited"]),
                        "min_ref_valid_for_spec_comparison": False,
                        "min_ref_tau_constraint_s": float(self.BI_REFERENCE_TAU_MIN_S),
                        "min_ref_min_terms_constraint": int(self.BI_REFERENCE_MIN_TERMS),
                        "min_ref_reason": min_ref_result["reason"],
                        "bs_10s_value_input_unit": float(bs_value),
                    })
                except Exception as exc:
                    print(f"{names[i]} Allan偏差计算失败: {exc}")
                    ax.text(0.5, 0.5, f"计算失败\n{exc}", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)

            conv = (
                "ADEV坐标单位与输入速率相同；RW系数单位为输入速率×√s。\n"
                "每√h换算=×60；仅在白速率噪声及0..fs/2单边PSD约定下，等效单边ASD=√2×RW，且不是直接PSD估计。\n"
                "正式BI只由连续近零斜率平台计算：B=σ平台/0.66428；最低ADEV折算值只是参考，不用于规格判定。\n"
                "10 s BS为非重叠分段均值总体标准差；平台与最低点门槛为本项目工程判据，不是标准强制值。\n"
                "n_terms 20/5分级仅为本项目工程可视化门槛，非标准规定。"
            )
            fig.text(0.5, -0.02, conv, ha="center", va="top", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5dc", alpha=0.95))
            plt.tight_layout(rect=[0, 0.09, 1, 1])
            save_path = os.path.join(self.save_dir, "03_零偏稳定性分析图.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
            self.report_data["加速度计零偏不稳定性_BI"] = acc_results
            self.report_data["陀螺仪零偏不稳定性_BI"] = gyro_results
            self.report_data["加速度计零偏稳定性"] = acc_bias
            self.report_data["陀螺仪零偏稳定性"] = gyro_bias
            self.report_data["Allan曲线数据"] = curve_path
            self.report_data["Allan参数汇总"] = summary_path
            self.report_data["Allan方差图"] = save_path  # 兼容旧键
            self.report_data["Allan偏差图"] = save_path
            return save_path
        except Exception as exc:
            print(f"Allan偏差图绘制失败: {exc}")
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
        """绘制长期零漂趋势图（基于欧拉角 Roll/Pitch/Yaw）"""
        try:
            # 行业通用做法：静态零漂后处理中，设备已输出欧拉角时直接由欧拉角构造
            # 旋转矩阵（欧拉角→DCM 与四元数→DCM 数学等价，无需先转四元数）
            euler_cols = ["roll", "pitch", "yaw"]
            if not all(col in self.df.columns for col in euler_cols):
                self.report_data["漂移分析图"] = "未检测到欧拉角数据"
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
            euler = self.df[euler_cols].values

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
                        r = R.from_euler("xyz", euler[start:end], degrees=True)
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
                # 行业通用做法：直接用设备输出的欧拉角构造旋转矩阵
                # （欧拉角→DCM 与四元数→DCM 数学等价，无需先转四元数）
                euler_cols = ["roll", "pitch", "yaw"]
                if not all(col in self.df.columns for col in euler_cols):
                    self.report_data["落点半径对比图"] = "未检测到欧拉角数据"
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
                euler = self.df[euler_cols].values
                results = {sec: [] for sec in window_seconds}
                
                for sec in window_seconds:
                    window_samples = sec * fs_int
                    if len(self.df) < window_samples + 100:
                        continue
                    
                    for start in range(0, len(self.df) - window_samples, step_size):
                        end = start + window_samples
                        try:
                            r = R.from_euler("xyz", euler[start:end], degrees=True)
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
        """把 Allan 结构化结果格式化为 HTML/Markdown 共用单元格。"""
        key = "加速度计零偏不稳定性_BI" if sensor == "acc" else "陀螺仪零偏不稳定性_BI"
        entry = self.report_data.get(key, {}).get(axis, {})
        if not entry:
            old_key = ("加速度计零偏稳定性_10s平滑" if sensor == "acc"
                       else "陀螺仪零偏稳定性_10s平滑")
            return {"rw": "N/A", "bi": "N/A", "min_ref": "N/A",
                    "bs": self.report_data.get(old_key, {}).get(axis, "N/A"),
                    "method": "N/A"}

        def fmt(value, unit):
            try:
                return ("N/A" if value is None or not np.isfinite(value)
                        else f"{value:.6e} {unit}")
            except (TypeError, ValueError):
                return "N/A"

        rw_result, bi_result = entry.get("rw", {}), entry.get("bi", {})
        min_result = entry.get("min_ref", {})
        rw_u, bi_u = entry.get("rw_units", {}), entry.get("bi_units", {})
        min_u, bs_u = entry.get("min_ref_units", {}), entry.get("bs_units", {})
        if sensor == "acc":
            rw = "<br>".join([
                fmt(rw_u.get("m/s/√s"), "m/s/√s"),
                fmt(rw_u.get("m/s/√h"), "m/s/√h"),
                fmt(rw_u.get("μg·√s"), "μg·√s"),
                "单边白噪声ASD等效：" + fmt(rw_u.get("单边ASD_μg/√Hz"), "μg/√Hz")])
            bi = "<br>".join([fmt(bi_u.get("m/s²"), "m/s²"),
                                fmt(bi_u.get("mg"), "mg"), fmt(bi_u.get("μg"), "μg")])
            min_ref = "<br>".join([fmt(min_u.get("m/s²"), "m/s²"),
                                     fmt(min_u.get("mg"), "mg"), fmt(min_u.get("μg"), "μg")])
            bs = "<br>".join([fmt(bs_u.get("m/s²"), "m/s²"),
                                fmt(bs_u.get("mg"), "mg"), fmt(bs_u.get("μg"), "μg")])
        else:
            rw = "<br>".join([fmt(rw_u.get("°/√s"), "°/√s"),
                                fmt(rw_u.get("°/√h"), "°/√h"),
                                fmt(rw_u.get("rad/√h"), "rad/√h"),
                                "单边白噪声ASD等效：" +
                                fmt(rw_u.get("单边ASD_(°/s)/√Hz"), "(°/s)/√Hz")])
            bi = "<br>".join([fmt(bi_u.get("°/s"), "°/s"), fmt(bi_u.get("°/h"), "°/h"),
                                fmt(bi_u.get("rad/h"), "rad/h")])
            min_ref = "<br>".join([fmt(min_u.get("°/s"), "°/s"),
                                     fmt(min_u.get("°/h"), "°/h"),
                                     fmt(min_u.get("rad/h"), "rad/h")])
            bs = "<br>".join([fmt(bs_u.get("°/s"), "°/s"), fmt(bs_u.get("°/h"), "°/h"),
                                fmt(bs_u.get("rad/h"), "rad/h")])
        rw_cell = (f"有效（拟合斜率={rw_result.get('slope', np.nan):.3f}）<br>{rw}"
                   if rw_result.get("valid") else
                   f"N/A<br>{rw_result.get('reason', '无可靠白噪声拟合区')}")
        if bi_result.get("valid"):
            edge = "；候选边界受限" if bi_result.get("edge_limited") else ""
            bi_cell = (f"平台识别通过{edge}<br>{bi}<br>"
                       f"tau≈{bi_result.get('platform_tau', np.nan):.6g} s；"
                       f"斜率={bi_result.get('fit_slope', np.nan):.3f}；"
                       f"最小n_terms={bi_result.get('fit_min_n_terms', np.nan):.0f}<br>"
                       "规格可比性未确认；需另行匹配厂家测试条件与定义")
        else:
            bi_cell = f"N/A<br>{bi_result.get('reason', '未找到可信平台')}"
        if min_result.get("available"):
            slope = min_result.get("local_slope", np.nan)
            slope_text = f"{slope:.3f}" if np.isfinite(slope) else "N/A"
            edge = "；候选范围边界" if min_result.get("edge_limited") else ""
            min_cell = (f"参考值，非正式BI<br>{min_ref}<br>"
                        f"tau={min_result.get('tau', np.nan):.6g} s；"
                        f"n_terms={min_result.get('n_terms', np.nan):.0f}；"
                        f"局部斜率={slope_text}{edge}<br>不用于规格合格判定")
        else:
            min_cell = f"N/A<br>{min_result.get('reason', '受约束候选点不足')}"
        return {"rw": rw_cell, "bi": bi_cell, "min_ref": min_cell, "bs": bs,
                "method": self.report_data.get("Allan分析配置", "未记录")}
    
    def generate_html_report(self):
        """生成HTML报告"""
        try:
            html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMU数据分析报告 — 无锡凌思 LINS16460</title>
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
        <h1>IMU 数据分析报告 — 无锡凌思 LINS16460</h1>
        <p class="subtitle">Powered by IMU Analysis Toolkit | 无锡凌思 LINS16460</p>
"""
            
            html += "\n        <div class='info-box'>\n            <div class='info-title'>数据基本信息</div>\n            <div class='info-grid'>\n"
            
            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位",
                "Allan曲线数据", "Allan参数汇总", "时间序列图", "统计分布图",
                "Allan方差图", "Allan偏差图", "PSD图", "相关性图", "漂移分析图",
                "落点半径对比图", "落点半径对比数据",
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
                html += "                <tr><th>指标</th><th>传感器</th><th>方法/口径</th><th>X轴</th><th>Y轴</th><th>Z轴</th></tr>\n"
                for sensor, label in [("gyro", "陀螺仪"), ("acc", "加速度计")]:
                    cells = [self._allan_report_cells(sensor, axis) for axis in ["x", "y", "z"]]
                    html += f"                <tr><td><b>正式零偏不稳定性 BI</b></td><td>{label}</td><td>{cells[0]['method']}<br>B=σ平台/0.66428；连续近零斜率平台<br>门槛为本项目工程判据</td>"
                    html += "".join(f"<td>{c['bi']}</td>" for c in cells) + "</tr>\n"
                    html += f"                <tr><td><b>最低 ADEV 等效参考</b><br><span style='color:#b83280;'>非正式 BI</span></td><td>{label}</td><td>tau≥1 s、n_terms≥20；σmin/0.66428<br>不用于规格判定</td>"
                    html += "".join(f"<td>{c['min_ref']}</td>" for c in cells) + "</tr>\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    html += f"                <tr><td><b>{rw_label}</b></td><td>{label}</td><td>0.1～10 s斜率拟合</td>"
                    html += "".join(f"<td>{c['rw']}</td>" for c in cells) + "</tr>\n"
                    html += f"                <tr><td><b>10 s 分段均值标准差</b></td><td>{label}</td><td>非重叠分段；ddof=0</td>"
                    html += "".join(f"<td>{c['bs']}</td>" for c in cells) + "</tr>\n"
                html += "            </table>\n        </div>\n"

                # 指标定义说明
                html += """
        <div class='info-box'>
            <div class='info-title'>指标定义说明</div>
            <table class='stats-table' style='font-size:13px;'>
                <tr><th style='background:#4a5568;'>指标名称</th><th style='background:#4a5568;'>定义</th><th style='background:#4a5568;'>计算方法</th><th style='background:#4a5568;'>依据/性质</th></tr>
                <tr><td><b>零偏不稳定性 (BI)</b></td>
                    <td>Allan 偏差中的 flicker/pink rate-noise 平台系数</td>
                    <td>3个原曲线连续近零斜率点；B=σ平台/0.66428；平台识别通过不等于规格可比</td>
                    <td>噪声模型关系与本项目工程判据；非通用合规声明</td></tr>
                <tr><td><b>最低 ADEV 等效参考</b></td>
                    <td>可信平台不明显时的受约束工程参考；不是 BI</td>
                    <td>tau≥1 s且n_terms≥20的最低ADEV/0.66428；报告支持项和边界状态</td>
                    <td>本项目工程判据；不用于规格合格判定</td></tr>
                <tr><td><b>10 s 分段均值标准差</b></td>
                    <td>10秒非重叠窗口均值的总体标准差</td>
                    <td>按10 s分段后计算 std(ddof=0)</td>
                    <td>工程统计量，条款适用性需另行核对</td></tr>
                <tr><td><b>角度随机游走 (ARW)</b></td>
                    <td>陀螺仪白噪声引起的角度积分随机游走</td>
                    <td>ADEV在0.1～10 s进行log-log斜率拟合，外推至1 s</td>
                    <td>Allan噪声模型参考</td></tr>
                <tr><td><b>速度随机游走 (VRW)</b></td>
                    <td>加速度计白噪声引起的速度积分随机游走</td>
                    <td>ADEV在0.1～10 s进行log-log斜率拟合，外推至1 s</td>
                    <td>Allan噪声模型参考</td></tr>
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
                <tr><td><b>加速度计 VRW 系数 N</b></td><td>m/s/√s（即(m/s²)·√s）</td><td>N×60</td><td>m/s/√h</td></tr>
                <tr><td><b>加速度计 BI/参考/BS</b></td><td>m/s²</td><td>×1e3/9.80665；×1e6/9.80665</td><td>mg；μg</td></tr>
                <tr><td><b>陀螺仪 ARW 系数 N</b></td><td>°/√s</td><td>N×60；N×π/180</td><td>°/√h；rad/√s</td></tr>
                <tr><td><b>陀螺仪 BI/参考/BS</b></td><td>°/s</td><td>×3600；×20π</td><td>°/h；rad/h</td></tr>
                <tr><td><b>条件等效单边白噪声 ASD</b></td><td>仅限白速率噪声和0..fs/2单边PSD约定</td><td>√2×N</td><td>(输入速率单位)/√Hz</td></tr>
            </table>
            <div style='margin-top:10px; padding:10px; background:#f0f4f8; border-left:4px solid #667eea; font-size:13px;'>
                <b>说明：</b><br>
                1. ADEV单位与输入速率序列相同；RW系数的单位是输入速率单位×√s。<br>
                2. BI仅对连续近零斜率平台计算，B=σ平台/0.66428；无平台时为N/A。<br>
                3. 最低ADEV等效参考不是正式BI，不用于规格判定。<br>
                4. 10 s BS是非重叠分段均值总体标准差(ddof=0)这一工程统计量。<br>
                5. √2×N是条件等效，不是直接PSD估计，也不自动等同厂家噪声密度。
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
            md = "# IMU 数据分析报告 — 无锡凌思 LINS16460\n\n"
            md += "---\n\n## 数据基本信息\n\n"
            
            hidden_report_keys = {
                "统计摘要", "加速度计零偏稳定性", "陀螺仪零偏稳定性",
                "加速度计零偏不稳定性_BI", "陀螺仪零偏不稳定性_BI",
                "加速度计零偏稳定性_10s平滑", "陀螺仪零偏稳定性_10s平滑",
                "加速度计BS_10s单位", "陀螺仪BS_10s单位", "Allan曲线数据",
                "Allan参数汇总", "时间序列图", "统计分布图", "Allan方差图",
                "Allan偏差图", "PSD图", "相关性图", "漂移分析图",
                "落点半径对比图", "落点半径对比数据",
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
                    md += f"| **正式零偏不稳定性 BI** | {label} | B=σ平台/0.66428；连续近零斜率平台；工程判据 |"
                    md += "".join(f" {c['bi']} |" for c in cells) + "\n"
                    md += f"| **最低 ADEV 等效参考（非正式BI）** | {label} | tau≥1 s、n_terms≥20；σmin/0.66428；不用于规格判定 |"
                    md += "".join(f" {c['min_ref']} |" for c in cells) + "\n"
                    rw_label = "VRW" if sensor == "acc" else "ARW"
                    md += f"| **{rw_label}** | {label} | 0.1～10 s斜率拟合 |"
                    md += "".join(f" {c['rw']} |" for c in cells) + "\n"
                    md += f"| **10 s 分段均值标准差** | {label} | 非重叠分段；ddof=0 |"
                    md += "".join(f" {c['bs']} |" for c in cells) + "\n"

                md += """
### 指标定义说明

| 指标名称 | 定义 | 计算方法 | 依据/性质 |
|----------|------|----------|----------|
| **零偏不稳定性 (BI)** | Allan偏差中的flicker/pink rate-noise平台系数 | 连续3点近零斜率平台；B=σ平台/0.66428；识别通过不等于规格可比 | 噪声模型关系与本项目工程判据；非通用合规声明 |
| **最低ADEV等效参考** | 可信平台不明显时的受约束工程参考；不是BI | tau≥1 s且n_terms≥20的最低ADEV/0.66428 | 本项目工程判据；不用于规格判定 |
| **10 s分段均值标准差** | 10秒非重叠窗口均值的总体标准差 | std(ddof=0) | 工程统计量，条款适用性需另行核对 |
| **角度随机游走 (ARW)** | 陀螺仪白噪声引起的角度积分随机游走 | ADEV在0.1～10 s拟合斜率并外推至1 s | Allan噪声模型参考 |
| **速度随机游走 (VRW)** | 加速度计白噪声引起的速度积分随机游走 | ADEV在0.1～10 s拟合斜率并外推至1 s | Allan噪声模型参考 |
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
| **加速度计 VRW系数N** | m/s/√s（即(m/s²)·√s） | N×60 | m/s/√h |
| **加速度计 BI/参考/BS** | m/s² | ×1e3/9.80665；×1e6/9.80665 | mg；μg |
| **陀螺仪 ARW系数N** | °/√s | N×60；N×π/180 | °/√h；rad/√s |
| **陀螺仪 BI/参考/BS** | °/s | ×3600；×20π | °/h；rad/h |
| **条件等效单边白噪声ASD** | 仅限白速率噪声和0..fs/2单边PSD约定 | √2×N | (输入速率单位)/√Hz |

**说明：**

1. ADEV单位与输入速率序列相同；RW系数单位是输入速率单位×√s。
2. BI仅对连续近零斜率平台计算，B=σ平台/0.66428；无平台时为N/A。
3. 最低ADEV等效参考不是正式BI，不用于规格判定。
4. 10 s BS是非重叠分段均值总体标准差(ddof=0)这一工程统计量。
5. √2×N是条件等效，不是直接PSD估计，也不自动等同厂家噪声密度。
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
            ("计算10s分段均值标准差", self.calculate_bias_stability_10s),
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
        self.root.title("IMU 数据分析工具箱 — 无锡凌思 LINS16460")
        self.root.geometry("800x600")
        
        self.file_path = tk.StringVar()
        self.sample_rate = tk.StringVar(value="")
        self.trim_minutes = tk.StringVar(value="0")
        self.analyzing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="IMU 数据分析工具箱 — 无锡凌思 LINS16460",
                                font=("Microsoft YaHei", 20, "bold"))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame,
                                  text="无锡凌思 LINS16460 一站式静态数据分析解决方案",
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
        ttk.Label(sr_frame, text="（必填，无默认值，例如 200）", font=("Microsoft YaHei", 9), foreground="#718096").pack(side=tk.LEFT)

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
            "✓ Allan偏差（ADEV）与零偏平台分析",
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
            filetypes=[("LINS16450/16460数据文件", "*.txt"), ("所有文件", "*.*")]
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
        
        # 验证采样率（必填，无默认值，由用户手动输入）
        sr_str = self.sample_rate.get().strip()
        if not sr_str:
            messagebox.showerror("错误", "采样率不能为空！\n请输入数据文件的采样率（单位：Hz），例如：200")
            return
        try:
            sr_val = float(sr_str)
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
