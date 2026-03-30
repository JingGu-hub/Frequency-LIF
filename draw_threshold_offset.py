import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.fft import fftfreq, fft
from scipy.interpolate import interpolate

from utils.utils import DFT_series_decomp, Series_decomp


def adaptive_kde(data, n_points=1000, bw_method='silverman', plot_bimodal=False):
    """
    自适应带宽的核密度估计

    Parameters:
    data: array-like, 输入数据
    n_points: int, 采样点数
    bw_method: str or scalar, 带宽选择方法
    plot_bimodal: bool, 是否检测并标记双峰

    Returns:
    x_range: array, x轴范围
    density: array, 对应的密度值
    (可选) peaks: array, 如果plot_bimodal=True，返回峰的位置
    """
    import numpy as np
    from scipy import stats
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    # 转换为numpy数组
    data = np.asarray(data)

    # 处理数据范围（添加一点边距，避免边缘效应）
    data_min, data_max = data.min(), data.max()
    margin = (data_max - data_min) * 0.05  # 5%的边距
    x_range = np.linspace(data_min - margin, data_max + margin, n_points)

    # 创建KDE对象（自适应带宽）
    kde = stats.gaussian_kde(data, bw_method=bw_method)

    # 计算密度
    density = kde(x_range)

    # 如果需要绘制双峰检测
    if plot_bimodal:
        # 检测峰
        peaks, properties = find_peaks(density, prominence=density.max() * 0.1)

        print(f"检测到 {len(peaks)} 个峰")

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制密度曲线
        ax.plot(x_range, density, color='navy', linewidth=2.5, label='KDE密度')

        # 标记所有检测到的峰
        if len(peaks) > 0:
            ax.plot(x_range[peaks], density[peaks], 'ro', markersize=8,
                    label=f'检测到的峰 (n={len(peaks)})')

            # 如果是双峰，特别标注
            if len(peaks) == 2:
                ax.plot(x_range[peaks], density[peaks], 'g*', markersize=12,
                        markeredgecolor='gold', markeredgewidth=2,
                        label='双峰特征 ✓')

                # 添加峰之间的连线（可选）
                ax.plot(x_range[peaks], density[peaks], 'g--', linewidth=1, alpha=0.5)

                print(f"双峰位置: x1={x_range[peaks[0]]:.2f}, x2={x_range[peaks[1]]:.2f}")

        # 添加直方图作为背景参考
        ax.hist(data, bins='auto', density=True, alpha=0.3, color='gray', label='数据直方图')

        # 设置标签和标题
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)

        # 根据峰的数量设置标题
        if len(peaks) == 2:
            ax.set_title('核密度估计 - 检测到双峰分布', fontsize=14, fontweight='bold')
        elif len(peaks) > 2:
            ax.set_title(f'核密度估计 - 检测到 {len(peaks)} 个峰', fontsize=14)
        else:
            ax.set_title('核密度估计 - 单峰分布', fontsize=14)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 返回x_range, density和peaks
        return x_range, density, peaks

    # 默认情况：只返回x_range和density
    return x_range, density, 0

def plot_membrane_distribution_auto(threshold_array, v_th=1.0, save_path=None):
    """
    绘制膜电位分布图，自动调整横纵轴范围

    参数:
    - threshold_array: numpy数组，神经元的膜电位值
    - v_th: 放电阈值，默认1.0
    - save_path: 保存路径
    """

    # 确保是一维数组
    data = np.array(threshold_array).flatten()

    # 去除无效值
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]

    # 自动计算横轴范围
    data_min = data.min()
    data_max = data.max()
    x_min = min(data_min, v_th - 1)  # 留出余量
    x_max = max(data_max, v_th + 1)
    # 扩展到整数边界
    x_min = np.floor(x_min)
    x_max = np.ceil(x_max)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 如果数据全为0（或全相同）
    if data.std() == 0:
        # 绘制垂直线
        unique_val = data[0]
        ax.axvline(x=unique_val, color='steelblue', linewidth=3,
                   linestyle='-', alpha=0.8, label=f'All values = {unique_val}')

        # 添加文本说明
        ax.text(0.5, 0.5, f'All membrane potentials = {unique_val}',
                transform=ax.transAxes, fontsize=14, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 设置纵轴范围
        ax.set_ylim(0, 1)

    else:
        # 正常数据：绘制直方图和密度曲线
        # 自动选择bins数量
        bins = min(50, len(np.unique(data)))

        # 绘制直方图
        n, bins, patches = ax.hist(data, bins=bins, density=True,
                                   alpha=0.7, color='steelblue',
                                   edgecolor='black', linewidth=0.8)

        # 绘制核密度估计曲线
        try:
            x_range, density = adaptive_kde(data, n_points=1000, bw_method='silverman', plot_bimodal=False)
            ax.plot(x_range, density, color='navy', linewidth=2.5)

            # 自动设置纵轴范围
            y_max = max(density.max(), n.max()) * 1.2
            ax.set_ylim(0, y_max)

        except:
            # 如果KDE失败，只用直方图
            ax.set_ylim(0, n.max() * 1.2)

    # 添加阈值线
    if v_th >= x_min and v_th <= x_max:
        ax.axvline(x=v_th, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(v_th + 0.1, ax.get_ylim()[1] * 0.9, f'$v_{{th}} = {v_th}$',
                color='red', fontsize=12, fontweight='bold')

    # 设置横轴范围
    ax.set_xlim(x_min, x_max)

    # 标题和标签
    ax.set_title('Membrane Potential Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Membrane Potential', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)

    # 网格
    ax.grid(True, alpha=0.3)

    # 显示统计信息
    stats_text = f'Mean: {np.mean(data):.2f}\nStd: {np.std(data):.2f}\nMin: {data_min:.2f}\nMax: {data_max:.2f}\nN: {len(data)}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def read_data(activate_type, dataset_name, predlen, apendix_s=None, epoch=0):
    if apendix_s is None:
        best_activation_out_name = './outputs/%s_results/%s/iTransformer_%s_predlen%d.npy' % (activate_type, dataset_name, dataset_name, predlen)
    else:
        best_activation_out_name = './outputs/%s_results/%s/iTransformer_%s_predlen%d_%s.npy' % (activate_type, dataset_name, dataset_name, predlen, apendix_s)

    best_activation_out = np.load(best_activation_out_name)

    return best_activation_out

def signal_to_freq_mag(signal, target_freqs=None, fs=10000, mode='full'):
    """
    将时域信号转换为频域幅值

    Parameters:
    -----------
    mode : str
        'full' : 返回完整频谱
        'fixed': 返回插值到target_freqs的频谱
    """
    if signal is None:
        return None, None if mode == 'full' else None

    # FFT
    N = len(signal)
    f = fftfreq(N, d=1 / fs)[:N // 2]
    mag = np.abs(fft(signal))[:N // 2]

    # 归一化
    if np.max(mag) > 0:
        mag = mag / np.max(mag)

    if mode == 'full':
        return f, mag
    else:  # fixed mode
        if target_freqs is None:
            target_freqs = np.array([100, 200, 300, 400, 500])
        f_interp = interpolate.interp1d(f, mag, bounds_error=False, fill_value=0)
        mag_target = f_interp(target_freqs)
        if np.max(mag_target) > 0:
            mag_target = mag_target / np.max(mag_target)
        return target_freqs, mag_target


def plot_spiking_spectrum(X_signal=None, R_signal=None, S_signal=None,
                          use_lpf=True, fs=10000, mode='full'):
    """
    传入时域信号，自动转频域并画图

    Parameters:
    -----------
    mode : str
        'full' : 画完整连续频谱（默认）
        'fixed': 画固定5个频点的stem图
    """

    # ============ 根据mode选择处理方式 ============
    if mode == 'full':
        # 完整频谱模式
        X_freqs, X_mag = signal_to_freq_mag(X_signal, fs=fs, mode='full')
        R_freqs, R_mag = signal_to_freq_mag(R_signal, fs=fs, mode='full')
        S_freqs, S_mag = signal_to_freq_mag(S_signal, fs=fs, mode='full')

        # 获取参考频率轴
        ref_freqs = None
        for f in [X_freqs, R_freqs, S_freqs]:
            if f is not None:
                ref_freqs = f
                break

        if ref_freqs is None:
            raise ValueError("至少需要传入一个信号")

        # 低通滤波器系数
        lpf_coef = 1 / (1 + (ref_freqs / 150) ** 2)

        # 创建画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：原始频谱
        if X_mag is not None:
            ax1.plot(X_freqs, X_mag, 'b-', label='X(f)', linewidth=1.5, alpha=0.8)
        if R_mag is not None:
            ax1.plot(R_freqs, R_mag, 'r-', label='R(f)', linewidth=1.5, alpha=0.8)
        if S_mag is not None:
            ax1.plot(S_freqs, S_mag, 'g-', label='S(f)', linewidth=1.5, alpha=0.8)

        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Normalized Magnitude', fontsize=11)
        ax1.set_title('(b) Frequency Spectrum (Full)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.set_xlim(0, 600)
        ax1.set_ylim(0, 1.2)

        # 右图：线性变换后
        right_title = '(c) After Linear Transform'
        if R_mag is not None:
            if use_lpf:
                R_conv = R_mag * lpf_coef[:len(R_mag)]
                right_title += ' (with LPF)'
            else:
                R_conv = R_mag
                right_title += ' (w/o LPF)'
            ax2.plot(R_freqs, R_conv, 'r-', label='ReLU + Conv', linewidth=1.5, alpha=0.8)

        if S_mag is not None:
            if use_lpf:
                S_conv = S_mag * lpf_coef[:len(S_mag)] * 0.6
            else:
                S_conv = S_mag * 0.6
            ax2.plot(S_freqs, S_conv, 'g-', label='Spiking + Conv', linewidth=1.5, alpha=0.8)

        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('Normalized Magnitude', fontsize=11)
        ax2.set_title(right_title, fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.25, linestyle='--')
        ax2.set_xlim(0, 600)
        ax2.set_ylim(0, 1.2)

    else:  # mode == 'fixed'
        # 固定频点模式（你原来的代码）
        freqs = np.array([100, 200, 300, 400, 500])

        _, X_mag = signal_to_freq_mag(X_signal, freqs, fs, mode='fixed')
        _, R_mag = signal_to_freq_mag(R_signal, freqs, fs, mode='fixed')
        _, S_mag = signal_to_freq_mag(S_signal, freqs, fs, mode='fixed')

        lpf_coef = 1 / (1 + (freqs / 150) ** 2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        width = 8
        if X_mag is not None:
            ax1.stem(freqs - width, X_mag, linefmt='blue', markerfmt='bo', basefmt=' ', label='X(f)')
        if R_mag is not None:
            ax1.stem(freqs, R_mag, linefmt='red', markerfmt='ro', basefmt=' ', label='R(f)')
        if S_mag is not None:
            ax1.stem(freqs + width, S_mag, linefmt='green', markerfmt='go', basefmt=' ', label='S(f)')

        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Normalized Magnitude', fontsize=11)
        ax1.set_title('(b) Frequency Spectrum (Fixed)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.set_xticks(freqs)
        ax1.set_xlim(50, 550)

        right_title = '(c) After Linear Transform'
        if R_mag is not None:
            if use_lpf:
                R_conv = R_mag * lpf_coef
                right_title += ' (with LPF)'
            else:
                R_conv = R_mag
                right_title += ' (w/o LPF)'
            ax2.stem(freqs - width, R_conv, linefmt='red', markerfmt='ro', basefmt=' ', label='ReLU + Conv')

        if S_mag is not None:
            if use_lpf:
                S_conv = S_mag * lpf_coef * 0.6
            else:
                S_conv = S_mag * 0.6
            ax2.stem(freqs + width, S_conv, linefmt='green', markerfmt='go', basefmt=' ', label='Spiking + Conv')

        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('Normalized Magnitude', fontsize=11)
        ax2.set_title(right_title, fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.25, linestyle='--')
        ax2.set_xticks(freqs)
        ax2.set_xlim(50, 550)

    plt.tight_layout()
    plt.suptitle(f'Frequency Spectrum Analysis ({mode} mode)', fontsize=13, y=1.02)
    plt.show()


def draw_spectrum_fig(gelu_activation_out, lif_activation_out, t):
    gelu_activation_out_t = gelu_activation_out[:, 0, :, :7].mean(axis=-1)
    lif_activation_out_t = lif_activation_out[:, t, :, :7].mean(axis=-1)
    plot_spiking_spectrum(R_signal=gelu_activation_out_t[1, :], S_signal=lif_activation_out_t[1, :], use_lpf=False, mode='none')

def plot_curves(arr, title='5 Curves from (5, 96) Array',
                xlabel='Index (0 to 95)', ylabel='Value',
                figsize=(12, 6), grid=True, legend=True):
    plt.figure(figsize=figsize)

    for i in range(arr.shape[0]):
        plt.plot(arr[i], label=f'Curve {i + 1}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend()
    if grid:
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def flatten_clean_reshape(arr, fill_value=None):
    # 保存原始形状
    original_shape = arr.shape

    # 展平为一维
    flat = arr.flatten()

    # 创建有效值掩码（非inf且非nan）
    valid_mask = np.isfinite(flat)

    # 提取有效值
    valid_values = flat[valid_mask]

    if fill_value is None:
        # 返回1D有效值数组
        return valid_values
    else:
        # 用填充值重构原形状
        result = np.full(original_shape, fill_value, dtype=arr.dtype)
        result.flat[valid_mask] = valid_values
        return result

def series_decomp_analysis():
    activate_type = 'lif'
    dataset_name = 'ETTh1'
    predlen = 96
    T = 4 if activate_type != 'gelu' else 1

    target_out = read_data(activate_type, dataset_name, predlen, 'target')
    predict_out = read_data(activate_type, dataset_name, predlen, 'predict')

    target_out_0 = target_out[0, :, :].mean(axis=-1).reshape(1, -1)
    predict_out_0 = predict_out[0, :, :, :].mean(axis=-1)
    # curve_data = np.concatenate((target_out_0, predict_out_0), axis=0)

    # plot_curves(curve_data)

    target_out = np.expand_dims(target_out, axis=1)
    target_out = (target_out - target_out.min()) / (target_out.max() - target_out.min())
    predict_out = (predict_out - predict_out.min()) / (predict_out.max() - predict_out.min())
    series_decomp = Series_decomp()
    target_season, target_trend = series_decomp(target_out)
    predict_season, predict_trend = series_decomp(predict_out)

    target_season_0 = target_season[0, :, :].mean(axis=-1).reshape(1, -1)
    predict_season_0 = predict_season[0, :, :, :].mean(axis=-1)
    target_trend_0 = target_trend[0, :, :].mean(axis=-1).reshape(1, -1)
    predict_trend_0 = predict_trend[0, :, :, :].mean(axis=-1)
    # season_data = np.concatenate((target_season_0, predict_season_0), axis=0)
    # trend_data = np.concatenate((target_trend_0, predict_trend_0), axis=0)

    # plot_curves(season_data)
    # plot_curves(trend_data)

    # target_out = np.repeat(target_out, T, axis=1)
    # target_season, target_trend = np.repeat(target_season, T, axis=1), np.repeat(target_trend, T, axis=1)
    # mae =np.abs(predict_out - target_out).mean(axis=0).mean(axis=-1).mean(axis=-1)
    # season_mae = np.abs(predict_season - target_season).mean(axis=0).mean(axis=-1).mean(axis=-1)
    # trend_mae = np.abs(predict_trend - target_trend).mean(axis=0).mean(axis=-1).mean(axis=-1)
    # print('mae', mae, 'season_mae:', season_mae, 'trend_mae:', trend_mae)

    print(flatten_clean_reshape(target_season / target_out).mean())
    print(flatten_clean_reshape(target_trend / target_out).mean())
    print(flatten_clean_reshape(predict_season / predict_out).mean())
    print(flatten_clean_reshape(predict_trend / predict_out).mean())

    print()

def thread_analysis():
    activate_type = 'aei_lif'
    dataset_name = 'ETTh1'
    predlen = 96
    T = 4 if activate_type != 'gelu' else 1
    epochs = 5

    # for epoch in range(epochs):
    #     # gelu_activation = read_data('gelu', dataset_name, predlen)
    lif_activation = read_data(activate_type, dataset_name, predlen)

        # 绘制图像（使用原图的坐标范围）
        # for t in range(T):
        #     activation_out_t = best_activation_out[:, t]
        #     # bias_act = (activation_out_t - 1)
        #     mean_activation_out_t = np.mean(activation_out_t, axis=0)
        #     std_activation_out_t = np.std(activation_out_t, axis=0)
        #     print('mean_v_%d: %.4f, std_activation_out_%d: %.4f' % (t, mean_activation_out_t, t, std_activation_out_t))
        #     plot_membrane_distribution_auto(activation_out_t)

    series_decomp = Series_decomp()
    for t in range(T):
        # gelu_x_season, gelu_x_trend = series_decomp(gelu_activation)
        # lif_x_season, lif_x_trend = series_decomp(lif_activation)
        # draw_spectrum_fig(gelu_x_season, lif_x_season, t)
        # draw_spectrum_fig(gelu_x_trend, lif_x_trend, t)

        activation_out = lif_activation[:, t, :, :].mean(axis=-1).mean(axis=-1)
        plot_membrane_distribution_auto(activation_out)

    print()

def main():
    thread_analysis()

if __name__ == '__main__':
    main()


