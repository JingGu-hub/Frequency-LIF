import numpy as np
from scipy.fft import fft


class GradientRatioMonitor:
    def __init__(self):
        self.ratio_history = []

    def compute_ratios(self, model, low_freq_cutoff=0.1, high_freq_cutoff=0.5):
        ratios = []

        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_np = param.grad.detach().cpu().numpy()

                # === 修复点：根据维度处理 ===
                if grad_np.ndim == 1:  # 1D: bias之类
                    # 直接对1D数组做FFT
                    grad_fft = np.abs(fft(grad_np))
                    avg_spectrum = grad_fft

                elif grad_np.ndim == 2:  # 2D: Linear层 [out_features, in_features]
                    grad_fft = np.abs(fft(grad_np, axis=1))
                    avg_spectrum = grad_fft.mean(axis=0)

                elif grad_np.ndim == 4:  # 4D: Conv2d层 [out_channels, in_channels, h, w]
                    # 对空间维度做FFT
                    from scipy.fft import fft2
                    grad_fft = np.abs(fft2(grad_np, axes=(2, 3)))
                    # 平均所有维度
                    avg_spectrum = grad_fft.mean(axis=(0, 1, 2, 3))

                else:  # 其他维度，展平处理
                    grad_flat = grad_np.flatten()
                    grad_fft = np.abs(fft(grad_flat))
                    avg_spectrum = grad_fft

                # 分离频率
                n_freqs = len(avg_spectrum)
                if n_freqs > 1:
                    low_idx = int(n_freqs * low_freq_cutoff)
                    high_idx = int(n_freqs * (1 - high_freq_cutoff))

                    low_mag = avg_spectrum[:low_idx].mean() if low_idx > 0 else avg_spectrum[0]
                    high_mag = avg_spectrum[high_idx:].mean() if high_idx < n_freqs else avg_spectrum[-1]

                    ratio = low_mag / (high_mag + 1e-8)
                    ratios.append(ratio)

        # 平均所有层的ratio
        avg_ratio = np.mean(ratios) if ratios else 1.0

        # 更新历史
        self.ratio_history.append(avg_ratio)
        if len(self.ratio_history) > 10:
            self.ratio_history = self.ratio_history[-10:]

        # 判断是否学习高频
        learn_high_freq_flag = self._should_learn_high_freq(avg_ratio)

        return ratios, learn_high_freq_flag

    def _should_learn_high_freq(self, ratio, target=1.0, tolerance=0.1, patience=1):
        current_near = (ratio > target - tolerance)

        # 更新历史
        self.ratio_history.append(current_near)
        if len(self.ratio_history) > patience:
            self.ratio_history = self.ratio_history[-patience:]

        # 检查最近patience次是否都接近1
        if len(self.ratio_history) == patience and all(self.ratio_history):
            return True

        return False


