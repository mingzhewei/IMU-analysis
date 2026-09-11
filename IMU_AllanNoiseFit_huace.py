# -*- coding: utf-8 -*-
"""
IMU Allan 噪声分项拟合图（N / B / K / R 虚线分解，供应商风格）
=====================================================================

定位（重要）
------------
本模块是 IMU_Analysis_huace.py 的【独立附加程序】：
- 不修改、不替换主分析程序的任何代码；
- 复用主程序的 BIN 解析函数 parse_binary_file（只读复用）；
- ADEV 计算口径与主程序报告完全一致（allantools ADEV 经典非重叠、
  相同采样率、相同掐头去尾设置）。

参数口径（遵循 IEEE 952 / IEEE 1780 行业惯例，与主程序报告一致）
----------------------------------------------------------------
- N（ARW/VRW，角度/速度随机游走）：
  取白噪声平台区（0.1 ~ 10 s，-1/2 斜率段）拟合并外推到 tau = 1 s 的
  ADEV 值。与主程序 extract_random_walk_coefficient 的判据一致。
  【明确不采用】把 tau < 0.1 s 的前端数字滤波滚降驼峰归入 -1/2 段
  的做法（该做法会系统性高估 N 约 30~40%）。
- B（零偏不稳定性）：B = sigma_platform / 0.66428，其中
  sigma_platform 取 tau >= 1 s 且支持项充足区间的最低 ADEV
  （与供应商结果逐位吻合的口径）。0.66428 = sqrt(2*ln2/pi)。
- K（速率随机游走，+1/2 斜率）与 R（速率斜坡，+1 斜率）：
  在扣除 N、B 分量后的长 tau 残差上做分段斜率拟合，不做四参数
  无约束全局拟合（B/K/R 在有限记录长度下简并，全局拟合数值无意义）。
  K、R 受本次记录长度（约 25 min）限制，仅供形态参考。

曲线模型（Allan 偏差域，tau 以小时计）：
    sigma^2(tau) = N^2/tau + (0.66428*B)^2 + K^2*tau/3 + R^2*tau^2/2

输出
----
- 09_Allan噪声分项拟合图_NBKR.png（六轴子图，风格与主报告一致）
- 09_Allan噪声分项拟合参数.csv
- 09_Allan噪声分项拟合说明.md
以上文件写入主程序 analysis_results 目录，不改变既有编号 01~08。
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import allantools as at

# ----------------------------------------------------------------------
# 只读复用主程序：导入解析函数与中文字体配置，不修改其任何内容
# ----------------------------------------------------------------------
_MAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "IMU_Analysis_huace.py")


def _load_main_module():
    spec = importlib.util.spec_from_file_location("imu_main_huace", _MAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 常量与模型
# ----------------------------------------------------------------------
FLICKER_FACTOR = float(np.sqrt(2.0 * np.log(2.0) / np.pi))  # 0.66428
G0 = 9.80665

RW_FIT_TAU_MIN_S = 0.1   # 白噪声平台区下限（避开前端数字滤波滚降驼峰）
RW_FIT_TAU_MAX_S = 10.0  # 白噪声平台区上限（与主程序口径一致）
RW_SLOPE_TOL = 0.3       # |斜率+0.5| 容差（与主程序判据一致）
BI_TAU_MIN_S = 1.0       # 平台（最低 ADEV）搜索下限
BI_MIN_TERMS = 20        # 平台点支持项门槛（与主程序一致）
MIN_SAMPLES = 100


def noise_model_ad2(tau_h, N, B, K, R):
    """N/B/K/R 合成 Allan 方差（tau 以小时为单位）。"""
    t = np.asarray(tau_h, dtype=float)
    return (N ** 2 / t
            + (FLICKER_FACTOR * B) ** 2
            + K ** 2 * t / 3.0
            + R ** 2 * t ** 2 / 2.0)


def _log_slope(taus, ad):
    return np.polyfit(np.log10(taus), np.log10(ad), 1)[0]


# ----------------------------------------------------------------------
# 参数提取
# ----------------------------------------------------------------------
def extract_nbkr(taus_s, adev_plot, n_terms, is_gyro=False):
    """从 ADEV 曲线提取 N、B、K、R。单位：绘图单位制。

    taus_s     : Allan tau（秒）
    adev_plot  : 绘图单位下的 ADEV（陀螺 °/h；加计 ug）
    n_terms    : 每个 tau 点的支持项数

    返回 dict，含各系数、诊断信息、有效性标志。
    """
    res = {
        "N": np.nan, "N_draw": np.nan, "B": np.nan, "K": np.nan, "R": np.nan,
        "rw_slope": np.nan, "rw_valid": False,
        "platform_adev": np.nan, "platform_tau_s": np.nan,
        "k_tau_range": "", "r_tau_range": "",
        "note": "",
    }
    taus_s = np.asarray(taus_s, dtype=float)
    adev_plot = np.asarray(adev_plot, dtype=float)
    n_terms = np.asarray(n_terms, dtype=float)
    ok = np.isfinite(taus_s) & np.isfinite(adev_plot) & (taus_s > 0) & (adev_plot > 0)
    taus_s, adev_plot, n_terms = taus_s[ok], adev_plot[ok], n_terms[ok]
    if len(taus_s) < 10:
        res["note"] = "有效 ADEV 点不足"
        return res

    # ---- N：白噪声平台区（0.1~10 s）-1/2 拟合，外推 tau=1 s ----
    w = ((taus_s >= RW_FIT_TAU_MIN_S) & (taus_s <= RW_FIT_TAU_MAX_S)
         & (n_terms >= 3))
    if np.count_nonzero(w) >= 3:
        slope, intercept = np.polyfit(np.log10(taus_s[w]),
                                      np.log10(adev_plot[w]), 1)
        res["rw_slope"] = float(slope)
        # 拟合线在 tau = 1 s 的取值即为该白噪声系数（绘图单位/√s）。
        # 模型中 tau 以小时计：sigma = N / sqrt(tau_h)，而 tau_h(1 s) = 1/3600，
        # 所以绘图单位下 N（单位/√h）恰等于 sigma(1 s) 本身，无需再乘 60。
        sigma_at_1s_plot = float(10 ** intercept)
        # 绘图系数：sigma_plot(tau_h) = N_draw / sqrt(tau_h)
        res["N_draw"] = sigma_at_1s_plot
        # 标准单位 N：
        # - 陀螺绘图纵轴为 °/h，N_std(°/√h) = N_draw / 60
        # - 加计绘图纵轴为 μg，N_std(m/s/√h) = N_draw * G0 * 60e-6
        if is_gyro:
            res["N"] = sigma_at_1s_plot / 60.0
        else:
            res["N"] = sigma_at_1s_plot * G0 * 60.0e-6
        res["rw_valid"] = bool(abs(slope + 0.5) <= RW_SLOPE_TOL)

    # ---- B：tau>=1 s 且支持项充足区间的最低 ADEV ----
    w = (taus_s >= BI_TAU_MIN_S) & (n_terms >= BI_MIN_TERMS)
    if np.count_nonzero(w) >= 1:
        idx_local = int(np.argmin(adev_plot[w]))
        idx = np.flatnonzero(w)[idx_local]
        res["platform_adev"] = float(adev_plot[idx])
        res["platform_tau_s"] = float(taus_s[idx])
        res["B"] = float(adev_plot[idx] / FLICKER_FACTOR)

    # ---- K：扣除 N、B 后，长 tau 段斜率约 +1/2 区间拟合 ----
    th = taus_s / 3600.0
    resid2 = adev_plot ** 2
    if np.isfinite(res["N"]):
        resid2 = resid2 - res["N"] ** 2 / th
    if np.isfinite(res["B"]):
        resid2 = resid2 - (FLICKER_FACTOR * res["B"]) ** 2
    resid = np.sqrt(np.clip(resid2, 0.0, None))
    resid_valid = resid2 > 0

    def _fit_segment(tau_lo, tau_hi, target_slope, slope_tol=0.45):
        m = (resid_valid & (taus_s >= tau_lo) & (taus_s <= tau_hi)
             & (n_terms >= 10))
        if np.count_nonzero(m) < 3:
            return np.nan, np.nan, "", np.nan
        sl = _log_slope(taus_s[m], resid[m])
        if abs(sl - target_slope) > slope_tol:
            return np.nan, sl, "", np.nan
        # 在该区间中点取残差幅度反推系数
        tmid = np.sqrt(taus_s[m].min() * taus_s[m].max())
        amid = float(np.exp(np.interp(np.log(tmid), np.log(taus_s[m]),
                                      np.log(resid[m]))))
        rng = "%.1f~%.1f s" % (taus_s[m].min(), taus_s[m].max())
        return amid, sl, rng, tmid

    # K: sigma_resid = K*sqrt(tau_h/3) -> K = sigma*sqrt(3/tau_h)
    amid, sl, rng, tmid = _fit_segment(30.0, min(200.0, taus_s.max() * 0.35), 0.5)
    if np.isfinite(amid):
        res["K"] = float(amid * np.sqrt(3.0 / (tmid / 3600.0)))
        res["k_tau_range"] = rng + ("（斜率 %.2f）" % sl)

    # R: sigma_resid = R*tau_h/sqrt(2) -> R = sigma*sqrt(2)/tau_h
    t_hi = taus_s.max()
    amid, sl, rng, tmid = _fit_segment(max(150.0, t_hi * 0.3), t_hi, 1.0)
    if np.isfinite(amid):
        res["R"] = float(amid * np.sqrt(2.0) / (tmid / 3600.0))
        res["r_tau_range"] = rng + ("（斜率 %.2f）" % sl)

    if not np.isfinite(res["K"]) and not np.isfinite(res["R"]):
        res["note"] = "记录长度内未识别出独立 K/R 斜率区（长 tau 段以上翘合项为主）"
    elif not np.isfinite(res["K"]):
        res["note"] = "K 未单独识别（与 B/R 区段重叠）；K/R 受记录长度限制仅供参考"
    elif not np.isfinite(res["R"]):
        res["note"] = "R 未单独识别（记录末端支持项不足）；K/R 受记录长度限制仅供参考"
    else:
        res["note"] = "K/R 受记录长度（约 25 min）限制，仅供形态参考"
    return res


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_nbkr_figure(curves, params, meta, out_png):
    """curves: dict[col] = (taus_s, adev_plot, n_terms)
    params : dict[col] = extract_nbkr 结果
    meta   : dict 含 title 所需单位与采样信息
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.08,
                        wspace=0.20, hspace=0.42)
    order = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    axis_cn = {"x": "X轴", "y": "Y轴", "z": "Z轴"}

    for i, col in enumerate(order):
        ax = axes[i // 3][i % 3]
        taus_s, adev_plot, ns = curves[col]
        p = params[col]
        sensor, axname = col.split("_")
        is_gyro = sensor == "gyro"
        unit = meta["gyro_plot_unit"] if is_gyro else meta["acc_plot_unit"]
        Nunit = "°/h$^{1/2}$" if is_gyro else "m/s/h$^{1/2}$"
        Bunit = "°/h" if is_gyro else "ug"
        Kunit = "°/h$^{1/2}$" if is_gyro else "(mm/s$^2$)/h$^{1/2}$"
        Runit = "°/h$^2$" if is_gyro else "(mm/s$^2$)/h"
        # K/R 显示换算（加计绘图单位为 ug）
        Kdisp = p["K"] * 9.80665e-3 if (not is_gyro and np.isfinite(p["K"])) else p["K"]
        Rdisp = p["R"] * 9.80665e-3 if (not is_gyro and np.isfinite(p["R"])) else p["R"]

        th = taus_s / 3600.0
        tgrid_h = np.logspace(np.log10(th.min()), np.log10(th.max()), 300)
        tgrid_s = tgrid_h * 3600.0

        # 拟合合成曲线（仅用有效系数）
        Nv = p["N_draw"] if np.isfinite(p.get("N_draw", np.nan)) else 0.0
        Bv = p["B"] if np.isfinite(p["B"]) else 0.0
        Kv = p["K"] if np.isfinite(p["K"]) else 0.0
        Rv = p["R"] if np.isfinite(p["R"]) else 0.0
        fit = np.sqrt(noise_model_ad2(tgrid_h, Nv, Bv, Kv, Rv))

        ax.loglog(tgrid_s, fit, color="lime", lw=2.2,
                  label="AVAR fitted")
        ax.plot(taus_s, adev_plot, linestyle="none", marker="o", ms=4.5,
                markerfacecolor="magenta", markeredgecolor="magenta",
                label="AVAR original")
        # 分项虚线（IEEE 952/1780 风格）
        if np.isfinite(p.get("N_draw", np.nan)):
            ax.loglog(tgrid_s, p["N_draw"] / np.sqrt(tgrid_h), "--",
                      color="0.45", lw=1.3, label="N")
        if np.isfinite(p["B"]):
            ax.loglog(tgrid_s, np.full_like(tgrid_s, FLICKER_FACTOR * p["B"]),
                      "--", color="0.45", lw=1.3, label="B")
        if np.isfinite(p["K"]):
            ax.loglog(tgrid_s, p["K"] * np.sqrt(tgrid_h / 3.0), "--",
                      color="0.45", lw=1.3, label="K")
        if np.isfinite(p["R"]):
            ax.loglog(tgrid_s, p["R"] * tgrid_h / np.sqrt(2.0), "--",
                      color="0.45", lw=1.3, label="R")

        # 标注滤波器滚降区与白噪声拟合区
        ax.axvspan(taus_s.min(), RW_FIT_TAU_MIN_S, color="0.9", alpha=0.55,
                   zorder=0, label="数字滤波滚降区(不拟合N)")
        ax.axvspan(RW_FIT_TAU_MIN_S, RW_FIT_TAU_MAX_S, color="#d8f5d8",
                   alpha=0.55, zorder=0, label="N拟合区0.1~10 s")

        title = ("%s%s  N=%.6f (%s); B=%.4f (%s); K=%s (%s); R=%s (%s)" % (
            ("陀螺仪" if is_gyro else "加速度计"), axis_cn[axname],
            p["N"], Nunit, p["B"], Bunit,
            ("%.4f" % Kdisp) if np.isfinite(Kdisp) else "N/A", Kunit,
            ("%.4f" % Rdisp) if np.isfinite(Rdisp) else "N/A", Runit))
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"$\tau$ / s")
        ax.set_ylabel(r"$\sigma_A$ / ( %s )" % unit)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Allan 噪声分项拟合（N/B/K/R）  数据=%s  采样率=%.0f Hz  时长=%.1f s  "
        "口径：N 取 0.1~10 s 白噪声平台区外推 tau=1 s（IEEE 952/1780 惯例）"
        % (meta["data_file"], meta["fs"], meta["duration_s"]),
        fontsize=13)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 说明文档
# ----------------------------------------------------------------------
def write_notes_md(params, meta, out_md):
    lines = [
        "# 09 Allan 噪声分项拟合（N/B/K/R）说明",
        "",
        "- 生成程序：`IMU_AllanNoiseFit_huace.py`（独立附加程序，未修改主分析程序）",
        "- 数据文件：`%s`" % meta["data_file"],
        "- 采样率：%.0f Hz（设备时间戳推定一致）；分析时长 %.1f s；掐头去尾 %.1f min（与主程序设置一致）" % (
            meta["fs"], meta["duration_s"], meta["trim_minutes"]),
        "- ADEV 口径：allantools 经典非重叠 ADEV（与主程序报告一致）",
        "",
        "## 参数口径",
        "",
        "- **N（ARW/VRW）**：白噪声平台区 0.1~10 s（-1/2 斜率段）拟合并外推到 tau = 1 s，"
        "符合 IEEE 952 / IEEE 1780 行业惯例，与主程序 `extract_random_walk_coefficient` 判据一致。",
        "- **B（零偏不稳定性）**：B = sigma_platform / 0.66428，sigma_platform 取 tau >= 1 s、"
        "n_terms >= 20 区间的最低 ADEV。",
        "- **K（速率随机游走，+1/2 斜率）/ R（速率斜坡，+1 斜率）**：扣除 N、B 分量后在长 tau "
        "残差上做分段斜率拟合；受本次约 25 min 记录长度限制，K/R 仅供形态参考，不用于规格判定。",
        "- 图中 tau < 0.1 s 灰色区域为传感器前端数字滤波滚降区，**不参与** N 的拟合"
        "（若把该驼峰误归入 -1/2 白噪声段，N 会被系统性高估约 30~40%）。",
        "",
        "## 拟合参数汇总",
        "",
        "| 轴 | N | B | K | R | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    name_cn = {"acc_x": "加速度计X轴", "acc_y": "加速度计Y轴", "acc_z": "加速度计Z轴",
               "gyro_x": "陀螺仪X轴", "gyro_y": "陀螺仪Y轴", "gyro_z": "陀螺仪Z轴"}
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        p = params[col]
        is_gyro = col.startswith("gyro")
        Nu = "°/√h" if is_gyro else "m/s/√h"
        Bu = "°/h" if is_gyro else "μg"
        Ku = "°/√h" if is_gyro else "(mm/s²)/√h"
        Ru = "°/h²" if is_gyro else "(mm/s²)/h"
        Kd = p["K"] * 9.80665e-3 if (not is_gyro and np.isfinite(p["K"])) else p["K"]
        Rd = p["R"] * 9.80665e-3 if (not is_gyro and np.isfinite(p["R"])) else p["R"]
        fmt = lambda v: ("%.6f" % v) if np.isfinite(v) else "N/A"
        fmt4 = lambda v: ("%.4f" % v) if np.isfinite(v) else "N/A"
        lines.append("| %s | %s %s | %s %s | %s %s | %s %s | %s |" % (
            name_cn[col], fmt(p["N"]), Nu, fmt4(p["B"]), Bu,
            fmt4(Kd), Ku, fmt4(Rd), Ru, p["note"]))
    lines += [
        "",
        "## 与主程序报告的关系",
        "",
        "- N 与主程序报告中的 ARW/VRW 为同一口径，数值应在拟合容差内一致；",
        "- B 与主程序“最低 ADEV 等效参考”同源（B = 等效参考值）；",
        "- 主程序报告的正式 BI（连续平台判据）门槛更严格，若主程序 BI 为 N/A 而本图为数值，"
        "属判据严格性差异，并非数据矛盾。",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def generate_nbkr_plot(bin_path, out_dir, sample_rate=500.0, trim_minutes=1.0):
    main = _load_main_module()
    df = main.parse_binary_file(bin_path)
    if df.empty:
        raise RuntimeError("BIN 解析结果为空：%s" % bin_path)
    fs = float(sample_rate)
    if trim_minutes and trim_minutes > 0:
        n_trim = int(trim_minutes * 60.0 * fs)
        if 2 * n_trim < len(df):
            df = df.iloc[n_trim:len(df) - n_trim].reset_index(drop=True)

    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    curves, params = {}, {}
    for col in cols:
        x = df[col].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < MIN_SAMPLES:
            continue
        taus, adev, _err, ns = at.adev(x, rate=fs, data_type="freq", taus="octave")
        is_gyro = col.startswith("gyro")
        adev_plot = adev * 3600.0 if is_gyro else adev * 1e6 / G0
        curves[col] = (np.asarray(taus, float), np.asarray(adev_plot, float),
                       np.asarray(ns, float))
        params[col] = extract_nbkr(*curves[col], is_gyro=is_gyro)

    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "data_file": os.path.basename(bin_path),
        "fs": fs,
        "duration_s": len(df) / fs,
        "trim_minutes": trim_minutes,
        "gyro_plot_unit": "°/h",
        "acc_plot_unit": "ug",
    }
    out_png = os.path.join(out_dir, "09_Allan噪声分项拟合图_NBKR.png")
    out_csv = os.path.join(out_dir, "09_Allan噪声分项拟合参数.csv")
    out_md = os.path.join(out_dir, "09_Allan噪声分项拟合说明.md")

    plot_nbkr_figure(curves, params, meta, out_png)

    rows = []
    for col in cols:
        if col not in params:
            continue
        p = dict(params[col])
        p.insert if False else None
        rows.append({
            "axis": col,
            "N_standard": p["N"],
            "N_draw_coefficient_plot_units": p.get("N_draw", np.nan),
            "N_unit": "°/√h" if col.startswith("gyro") else "m/s/√h",
            "B_plot_unit": p["B"],
            "B_unit": "°/h" if col.startswith("gyro") else "ug",
            "K_plot_unit": p["K"],
            "K_display_unit": p["K"] * 9.80665e-3 if (not col.startswith("gyro") and np.isfinite(p["K"])) else p["K"],
            "K_unit": "°/√h" if col.startswith("gyro") else "(mm/s^2)/√h",
            "R_plot_unit": p["R"],
            "R_display_unit": p["R"] * 9.80665e-3 if (not col.startswith("gyro") and np.isfinite(p["R"])) else p["R"],
            "R_unit": "°/h^2" if col.startswith("gyro") else "(mm/s^2)/h",
            "rw_slope_0p1_10s": p["rw_slope"],
            "rw_valid": p["rw_valid"],
            "platform_adev": p["platform_adev"],
            "platform_tau_s": p["platform_tau_s"],
            "k_tau_range": p["k_tau_range"],
            "r_tau_range": p["r_tau_range"],
            "note": p["note"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_notes_md(params, meta, out_md)
    return {"png": out_png, "csv": out_csv, "md": out_md, "params": params}


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(base, "huace_data", "COM7_20260910_114132.bin")
    results_dir = os.path.join(base, "huace_data", "analysis_results")
    if len(sys.argv) >= 2:
        bin_file = sys.argv[1]
    if len(sys.argv) >= 3:
        results_dir = sys.argv[2]
    outs = generate_nbkr_plot(bin_file, results_dir, sample_rate=500.0, trim_minutes=1.0)
    print("[09] 噪声分项拟合图:", outs["png"])
    print("[09] 拟合参数 CSV :", outs["csv"])
    print("[09] 说明文档     :", outs["md"])
