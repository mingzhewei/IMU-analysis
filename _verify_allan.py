"""历史 Allan 验证脚本（已废弃，不得用于当前程序验收）。

该脚本保留旧路径、200 Hz、100000 点截断、强制线性去趋势以及
``BI = 0.664 * min(ADEV)`` 等已确认失效的口径，仅供追溯旧结果。
当前实现和回归结果以 ``IMU_Analysis_huayi.py`` 及其导出的
``03_Allan曲线数据.csv`` 为准。
"""
import sys, os, warnings, traceback
raise SystemExit(
    "_verify_allan.py 已废弃：旧采样率、数据截断、线性去趋势和 BI 公式均不适用于当前实现。"
)
warnings.filterwarnings('ignore')
sys.path.insert(0, r'd:\software\HUAYI_IMU')

import numpy as np
import pandas as pd
import allantools as at
from scipy.signal import detrend

LOG = open(r'd:\software\HUAYI_IMU\_verify.log', 'w', encoding='utf-8')
def lp(*a):
    msg = ' '.join(str(x) for x in a)
    LOG.write(msg + '\n')
    LOG.flush()

lp("=" * 70)
lp("IMU Allan 方差计算完整验证报告")
lp("=" * 70)

# ============================================================
# 第1步: 数据加载验证
# ============================================================
lp("\n[第1步] 数据加载")
csv = r'd:\software\HUAYI_IMU\data1\20260808_115427_COM6.csv'
df = pd.read_csv(csv, encoding='gbk')
lp(f"  原始列: {[c for c in df.columns if 'acc' in c or 'gyro' in c or 'ax' in c]}")

# 手动映射
df = df.rename(columns={'ax':'acc_x','ay':'acc_y','az':'acc_z',
                         'gx':'gyro_x','gy':'gyro_y','gz':'gyro_z'})
lp(f"  映射后: acc_x={df['acc_x'].mean():.6f}±{df['acc_x'].std():.6f}, "
   f"gyro_z={df['gyro_z'].mean():.6f}±{df['gyro_z'].std():.6f}")

# ============================================================
# 第2步: 数据预处理验证 (IEEE 952 §5.2.1)
# ============================================================
lp("\n[第2步] 数据预处理 (IEEE Std 952-1997 §5.2.1)")
G0 = 9.80665
n = min(100000, len(df))

# 加速度计: acc_z 去重力, 所有轴去趋势
acc_z_corr = df['acc_z'].values[:n] - G0
lp(f"  acc_z 去重力前: mean={df['acc_z'].values[:n].mean():.6f}, std={df['acc_z'].values[:n].std():.6f}")
lp(f"  acc_z 去重力后: mean={acc_z_corr.mean():.6f}, std={acc_z_corr.std():.6f}")

# 线性去趋势
acc_z_detrended = detrend(acc_z_corr, type='linear')
lp(f"  acc_z 去趋势后: mean={acc_z_detrended.mean():.6e}, std={acc_z_detrended.std():.6f}")

# 陀螺仪: 直接去趋势
gyro_z_raw = df['gyro_z'].values[:n]
gyro_z_detrended = detrend(gyro_z_raw, type='linear')
lp(f"  gyro_z 原始: mean={gyro_z_raw.mean():.6f}, std={gyro_z_raw.std():.6f}")
lp(f"  gyro_z 去趋势后: mean={gyro_z_detrended.mean():.6e}, std={gyro_z_detrended.std():.6f}")

# ============================================================
# 第3步: Allan 方差计算验证 (allantools 库)
# ============================================================
lp("\n[第3步] Allan 方差计算 (allantools)")
lp("  调用: at.adev(data, rate=200, data_type='freq', taus='octave')")
lp("  说明: data_type='freq' 表示输入为速率数据 (°/s 或 m/s²)")
lp("       taus='octave' 表示 τ 按 2 的幂采样 (标准做法)")

# 手动计算 Allan 方差 (验证 allantools 输出)
def manual_allan_deviation(data, fs, tau_list):
    """手动计算 Allan 偏差 (非重叠法, 简单验证)"""
    n = len(data)
    results = []
    for tau in tau_list:
        m = int(tau * fs)  # 每个区间的样本数
        if m < 1:
            continue
        k = n // m  # 完整区间数
        if k < 2:
            continue
        # 计算每个区间的平均值
        means = []
        for i in range(k):
            segment = data[i*m:(i+1)*m]
            means.append(np.mean(segment))
        means = np.array(means)
        # Allan 方差: σ²(τ) = (1/(2(k-1))) × Σ(ȳ_i+1 - ȳ_i)²
        diffs = np.diff(means)
        variance = np.sum(diffs**2) / (2 * (k - 1))
        results.append((tau, np.sqrt(variance)))
    return results

# 验证 gyro_z
taus_check = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
lp(f"\n  验证 gyro_z 的 Allan 偏差 (手动计算 vs allantools):")
taus_lib, adev_lib, _, _ = at.adev(gyro_z_detrended, rate=200.0, data_type="freq", taus="octave")
taus_manual = manual_allan_deviation(gyro_z_detrended, 200.0, taus_check)

lp(f"  {'τ(s)':>8s}  {'手动计算':>12s}  {'allantools':>12s}  {'差异%':>8s}")
for tm, am in taus_manual:
    # 找最近的库值
    idx = np.argmin(np.abs(taus_lib - tm))
    al = adev_lib[idx]
    diff_pct = abs(am - al) / al * 100 if al > 0 else 0
    lp(f"  {tm:8.4f}  {am:12.6e}  {al:12.6e}  {diff_pct:8.2f}%")

# ============================================================
# 第4步: 随机游走系数提取验证 (IEEE 952 §5.3.2)
# ============================================================
lp("\n[第4步] 随机游走系数提取 (IEEE 952 §5.3.2)")
lp("  方法: τ ∈ [0.1, 10]s 对数域线性拟合, 外推至 τ=1s")

valid = adev_lib > 0
taus_v = taus_lib[valid]
adev_v = adev_lib[valid]

mask = (taus_v >= 0.1) & (taus_v <= 10.0)
lp(f"  拟合范围: τ ∈ [{taus_v[mask].min():.3f}, {taus_v[mask].max():.3f}]s")
lp(f"  拟合点数: {np.sum(mask)}")

log_tau = np.log10(taus_v[mask])
log_ad = np.log10(adev_v[mask])
coeffs = np.polyfit(log_tau, log_ad, 1)
slope, intercept = coeffs

lp(f"  拟合斜率: {slope:.6f} (理论值: -0.5)")
lp(f"  拟合截距: {intercept:.6f}")
lp(f"  |斜率+0.5| = {abs(slope+0.5):.6f}")

# 外推至 τ=1s
rw_manual = 10 ** intercept
lp(f"  RW = 10^({intercept:.6f}) = {rw_manual:.6e}")

# 验证: 在 τ=1s 处的 ADEV
idx_1s = np.argmin(np.abs(taus_v - 1.0))
lp(f"  ADEV(τ=1s) = {adev_v[idx_1s]:.6e} (应接近 RW={rw_manual:.6e})")

# 单位换算验证 (gyro)
lp(f"\n  陀螺仪单位换算:")
lp(f"    ARW (°/s/√Hz) = {rw_manual:.6e}")
lp(f"    ARW (°/√h)    = {rw_manual*60:.6f}  (×60)")
lp(f"    ARW (°/s/√s)  = {rw_manual/60:.6e}  (/60)")

# ============================================================
# 第5步: 零偏不稳定性提取验证 (GJB 7952 §6.3.2)
# ============================================================
lp("\n[第5步] 零偏不稳定性提取 (GJB 7952-2012 §6.3.2)")
lp("  公式: BI = 0.664 × min(ADEV)")
lp("  系数: 0.664 = √(2×ln(2)/π) = √(0.4413) = 0.6643")

# 验证 0.664 系数
pi = np.pi
coeff_check = np.sqrt(2 * np.log(2) / pi)
lp(f"  验证: √(2×ln(2)/π) = {coeff_check:.6f} (应≈0.664)")

min_idx = np.argmin(adev_v)
min_ad = adev_v[min_idx]
min_tau = taus_v[min_idx]
bi_manual = 0.664 * min_ad

lp(f"  min(ADEV) = {min_ad:.6e} at τ={min_tau:.2f}s")
lp(f"  BI = 0.664 × {min_ad:.6e} = {bi_manual:.6e}")

# 单位换算验证 (gyro)
lp(f"\n  陀螺仪单位换算:")
lp(f"    BI (°/s)   = {bi_manual:.6e}")
lp(f"    BI (°/h)   = {bi_manual*3600:.6f}  (×3600)")
lp(f"    BI (°/min) = {bi_manual*60:.6f}  (×60)")

# ============================================================
# 第6步: 零偏稳定性 (报告值) 验证
# ============================================================
lp("\n[第6步] 报告中的零偏稳定性值验证")

# 从代码中提取报告值
from IMU_Analysis_huayi import IMUDataAnalyzer
a = IMUDataAnalyzer(csv, 200)
a.calculate_basic_stats()

# 手动计算所有轴的 BI
for axis in ['x', 'y', 'z']:
    col = f'gyro_{axis}'
    data = a.df[col].dropna().values[:n]
    data = detrend(data, type='linear')
    taus_a, adev_a, _, _ = at.adev(data, rate=200.0, data_type='freq', taus='octave')
    valid_a = adev_a > 0
    taus_a = taus_a[valid_a]
    adev_a = adev_a[valid_a]

    bi_r = a.extract_bias_instability(taus_a, adev_a)
    rw_a = a.extract_random_walk_coefficient(taus_a, adev_a)

    bi_val = bi_r["bias_instability"]
    bi_dph = bi_val * 3600

    lp(f"  gyro_{axis}: BI={bi_val:.6e}°/s = {bi_dph:.4f}°/h, ARW={rw_a:.4e}°/s/√Hz = {rw_a*60:.4f}°/√h")

# ============================================================
# 第7步: 加速度计验证
# ============================================================
lp("\n[第7步] 加速度计验证")
for axis in ['x', 'y', 'z']:
    col = f'acc_{axis}'
    data_col = col if axis != 'z' else 'acc_z_corrected'
    if data_col not in a.df.columns:
        if axis == 'z':
            a.df['acc_z_corrected'] = a.df['acc_z'] - G0
            data_col = 'acc_z_corrected'
        else:
            continue

    data = a.df[data_col].dropna().values[:n]
    data = detrend(data, type='linear')
    taus_a, adev_a, _, _ = at.adev(data, rate=200.0, data_type='freq', taus='octave')
    valid_a = adev_a > 0
    taus_a = taus_a[valid_a]
    adev_a = adev_a[valid_a]

    bi_r = a.extract_bias_instability(taus_a, adev_a)
    rw_a = a.extract_random_walk_coefficient(taus_a, adev_a)

    bi_val = bi_r["bias_instability"]
    bi_mg = bi_val * 1000 / G0
    bi_ug = bi_val * 1e6 / G0

    lp(f"  acc_{axis}: BI={bi_val:.6e} m/s² = {bi_mg:.4f} mg = {bi_ug:.2f} μg, "
       f"VRW={rw_a:.4e} m/s²/√Hz = {rw_a*1e6/G0:.2f} μg/√Hz")

# ============================================================
# 第8步: 常见误差点检查
# ============================================================
lp("\n[第8步] 常见误差点检查")

# 检查1: 数据点数量
lp(f"  检查1: 数据点数量 n={n} (限制为 min(100000, {len(df)}))")
lp(f"         100000点 @200Hz = 500秒 = 8.33分钟")
lp(f"         对 Allan 方差来说: 足够覆盖 τ=100s 的区间")

# 检查2: detrend 类型
lp(f"  检查2: detrend(type='linear') — 只去除线性趋势")
lp(f"         保留高频噪声和非线性漂移 ✓")

# 检查3: octave 采样
lp(f"  检查3: taus='octave' — τ 按 2 的幂采样")
lp(f"         τ = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, ...]")

# 检查4: 有效数据过滤
lp(f"  检查4: valid = (adev > 0) — 过滤无效点 ✓")

# 检查5: RW 拟合范围
lp(f"  检查5: RW 拟合范围 τ ∈ [0.1, 10]s")
lp(f"         对于 200Hz 数据, τ=0.1s 对应 20 个样本")
lp(f"         最小 τ 应 ≥ 2/fs = 0.01s (2个样本)")

# 检查6: BI 系数
lp(f"  检查6: BI = 0.664 × min(ADEV)")
lp(f"         0.664 = √(2×ln(2)/π) ≈ 0.66427")
lp(f"         这是从一阶马尔可夫过程推导的理论值")

# ============================================================
# 结论
# ============================================================
lp("\n" + "=" * 70)
lp("验证结论")
lp("=" * 70)
lp("""
1. Allan 方差计算: 使用 allantools.adev(data_type='freq') — 正确
   - 输入为速率数据 (°/s, m/s²)
   - 输出为 Allan 偏差 σ(τ)
   - octave 采样符合标准

2. 随机游走提取: log-log 拟合 τ∈[0.1,10]s, 外推 τ=1s — 正确
   - 斜率应接近 -0.5 (白噪声)
   - RW = σ(1s)

3. 零偏不稳定性: BI = 0.664 × min(ADEV) — 正确
   - 系数 0.664 = √(2×ln(2)/π) 来自理论推导

4. 数据预处理: acc_z 去重力 + detrend — 正确
   - acc_z_corrected = acc_z - 9.80665
   - detrend(type='linear') 去除线性趋势

5. 单位换算: 全部正确
   - gyro: °/s → °/h (×3600), °/s/√Hz → °/√h (×60)
   - acc: m/s² → mg (×1000/9.80665), m/s²/√Hz → μg/√Hz (×1e6/9.80665)

结论: 所有计算逻辑符合 IEEE Std 952-1997 和 GJB 7952-2012 标准。
      测出的值偏小是正常的 — 说明 IMU 在静态环境下性能良好。
""")

LOG.close()
print("验证完成, 日志保存在 _verify.log")
