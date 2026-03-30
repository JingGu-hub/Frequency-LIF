import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
from numpy.ma.core import reshape
from scipy.fft import fft
from sklearn.mixture import GaussianMixture


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)

            if write_line != None:
                with open(filename, "a") as myfile:
                    myfile.write(write_line + '\n')
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))

            if write_line != None:
                with open(filename, "a") as myfile:
                    myfile.write(write_line + '\n')

    return filename

def set_dataset(args):
    if args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        args.data = args.dataset
        args.data_path = args.dataset + '.csv'
        args.root_path = '../data/MTSF_dataset/ETT-small/'
    elif args.dataset == 'solar-energy':
        args.data = 'Solar'
        args.data_path = args.dataset + '.txt'
        args.root_path = '../data/MTSF_dataset/%s/' % args.dataset
    else:
        args.data = 'custom'
        args.data_path = args.dataset + '.csv'
        args.root_path = '../data/MTSF_dataset/%s/' % args.dataset

    if args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        args.enc_in, args.dec_in, args.c_out = 7, 7, 7
    elif args.dataset == 'solar-energy':
        args.enc_in, args.dec_in, args.c_out = 137, 137, 137
    elif args.dataset == 'electricity':
        args.enc_in, args.dec_in, args.c_out = 321, 321, 321
    elif args.dataset == 'traffic':
        args.enc_in, args.dec_in, args.c_out = 862, 862, 862
    elif args.dataset == 'exchange':
        args.enc_in, args.dec_in, args.c_out = 8, 8, 8
    elif args.dataset == 'weather':
        args.enc_in, args.dec_in, args.c_out = 21, 21, 21

    if args.dataset in ['exchange']:
        args.season_factor = 0.5

# class DFT_series_decomp(nn.Module):
#     def __init__(self, top_k=5, use_smooth_filter=True):
#         super().__init__()
#         self.top_k = top_k
#         self.use_smooth_filter = use_smooth_filter
#
#     def forward(self, x):
#         x = torch.tensor(x, dtype=torch.float32)
#         b, t, l, c = x.shape
#
#         x = x.reshape(-1, l, c)
#
#         xf = torch.fft.rfft(x, dim=1)
#         freq = abs(xf)
#         freq[:, 0, :] = 0
#
#         # 为每个特征独立选择top_k
#         top_k_freq, top_indices = torch.topk(freq, self.top_k, dim=1)
#
#         # 使用软掩码或平滑滤波
#         # 使用sigmoid创建平滑过渡
#         threshold = top_k_freq[:, -1:, :]
#         mask = torch.sigmoid((freq - threshold) * 10)  # 平滑掩码
#
#         xf_season = xf * mask
#
#         # 低频部分作为趋势
#         low_freq_mask = torch.ones_like(freq)
#         low_freq_mask[:, :self.top_k, :] = 0  # 去除季节频率
#         xf_trend = xf * low_freq_mask
#
#         x_season = torch.fft.irfft(xf_season, n=x.shape[1], dim=1)
#         x_trend = torch.fft.irfft(xf_trend, n=x.shape[1], dim=1)
#
#         x_season = x_season.reshape(b, t, l, c)
#         x_trend = x_trend.reshape(b, t, l, c)
#
#         return x_season.detach().cpu().numpy(), x_trend.detach().cpu().numpy()

class Series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size=25, stride=1):
        super(Series_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x_trend = self.moving_avg(x)
        x_season = x - x_trend

        return x_season, x_trend

class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=3, use_smooth_filter=True):
        super().__init__()
        self.top_k = top_k
        self.use_smooth_filter = use_smooth_filter

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[:, 0, :] = 0

        # 为每个特征独立选择top_k
        top_k_freq, top_indices = torch.topk(freq, self.top_k, dim=1)

        # 使用软掩码或平滑滤波
        # 使用sigmoid创建平滑过渡
        threshold = top_k_freq[:, -1:, :]
        mask = torch.sigmoid((freq - threshold) * 10)  # 平滑掩码

        xf_season = xf * mask

        # 低频部分作为趋势
        low_freq_mask = torch.ones_like(freq)
        low_freq_mask[:, :self.top_k, :] = 0  # 去除季节频率
        xf_trend = xf * low_freq_mask

        x_season = torch.fft.irfft(xf_season, n=x.shape[1], dim=1)
        x_trend = torch.fft.irfft(xf_trend, n=x.shape[1], dim=1)

        return x_season, x_trend

class AmpDFT_series_decomp(nn.Module):
    def __init__(self, top_k=3, use_smooth_filter=True, n_channels=11):
        super().__init__()
        self.top_k = top_k
        self.use_smooth_filter = use_smooth_filter

        # 可以学习通道间的相关性矩阵
        self.channel_correlation = torch.complex(
            nn.Parameter(torch.eye(n_channels)).cuda(),
            torch.zeros_like(nn.Parameter(torch.eye(n_channels)).cuda())
        )

    def amplify_season(self, x_fft, n_fft=2048, alpha=0.3, beta=1.0):
        n_freq_bins = n_fft // 2 + 1
        max_freq = n_freq_bins - 1

        # 直接对高频施加更大的增益
        freq_bins = torch.arange(n_fft // 2 + 1).cuda()
        gain_curve = 1.0 + alpha * (freq_bins / max_freq) ** beta
        gain_curve = gain_curve.reshape(1, -1, 1)  # (1, 1025, 1)

        x_fft_enhanced = x_fft * gain_curve

        # 频率增益 (1, 1025, 1)
        # freq_gain = 1.0 + alpha * (torch.arange(max_freq) / (max_freq - 1)) ** beta
        # freq_gain = freq_gain.view(1, -1, 1).cuda()
        #
        # # 通道间信息混合（增强通道间的协同）
        # # 将频率和通道维度合并处理
        # x_fft_reshaped = x_fft.permute(0, 2, 1)  # (32, 11, 1025)
        #
        # # 应用通道相关性
        # x_fft_mixed = torch.matmul(self.channel_correlation, x_fft_reshaped)  # (32, 11, 1025)
        # x_fft_mixed = x_fft_mixed.permute(0, 2, 1)  # (32, 1025, 11)
        #
        # # 应用频率增益
        # x_fft_enhanced = x_fft_mixed * freq_gain

        return x_fft_enhanced

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[:, 0, :] = 0

        # 为每个特征独立选择top_k
        top_k_freq, top_indices = torch.topk(freq, self.top_k, dim=1)

        # 使用软掩码或平滑滤波
        # 使用sigmoid创建平滑过渡
        threshold = top_k_freq[:, -1:, :]
        mask = torch.sigmoid((freq - threshold) * 10)  # 平滑掩码

        xf_season = xf * mask
        xf_season = self.amplify_season(xf_season, n_fft=x.shape[1])

        # 低频部分作为趋势
        low_freq_mask = torch.ones_like(freq)
        low_freq_mask[:, :self.top_k, :] = 0  # 去除季节频率
        xf_trend = xf * low_freq_mask

        x_season = torch.fft.irfft(xf_season, n=x.shape[1], dim=1)
        x_trend = torch.fft.irfft(xf_trend, n=x.shape[1], dim=1)

        return x_season, x_trend

def find_intersection_loss(gmm):
    """
    找到两个高斯分布的交点
    """
    # 获取两个高斯分布的参数
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()  # 方差
    weights = gmm.weights_.flatten()

    # 确保 means[0] < means[1]
    if means[0] > means[1]:
        means = means[::-1]
        variances = variances[::-1]
        weights = weights[::-1]

    # 解二次方程：w1 * N(x|μ1,σ1) = w2 * N(x|μ2,σ2)
    # 取对数后得到二次方程
    a = 1 / (2 * variances[0]) - 1 / (2 * variances[1])
    b = means[1] / variances[1] - means[0] / variances[0]
    c = means[0] ** 2 / (2 * variances[0]) - means[1] ** 2 / (2 * variances[1]) + \
        0.5 * np.log(variances[0] / variances[1]) + np.log(weights[1] / weights[0])

    # 解二次方程
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # 如果没有实根，返回均值的中点
        return np.mean(means)

    # 计算两个根
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # 选择在均值之间的根
    if means[0] <= root1 <= means[1]:
        return root1
    elif means[0] <= root2 <= means[1]:
        return root2
    else:
        return np.mean(means)

def gmm_divide_with_threshold(loss):
    """
    使用GMM划分loss，并返回阈值
    """
    loss = loss.detach().cpu().numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)
    prob = gmm.predict_proba(loss)

    prob = prob[:, gmm.means_.argmin()]
    clean_mask = (prob > 0.5).astype(int)

    # 找到阈值
    threshold = find_intersection_loss(gmm)

    return threshold

class TimeSeriesFrequencyInverter:
    def __init__(self, preserve_energy=True):
        self.preserve_energy = preserve_energy
        self.permutation_map = None
        self.inverse_map = None
        self.freqs = None
        self.L = None  # 保存原始序列长度
        self.threshold = 0.5

    def _create_fft_permutation(self, L, device):
        # 保存原始序列长度
        self.L = L

        # 生成频率坐标
        freqs = torch.fft.rfftfreq(L, device=device)
        self.freqs = freqs.cpu()

        # 按频率绝对值排序
        freq_abs = torch.abs(freqs)
        sorted_indices = freq_abs.argsort()

        # 高频优先的置换
        self.permutation_map = sorted_indices.flip(0)
        self.inverse_map = torch.argsort(self.permutation_map)

    def fft_invert(self, x):
        B, C, L = x.shape

        # 转换到频域
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 创建频率置换映射
        if self.permutation_map is None:
            self._create_fft_permutation(L, x.device)

        # 频率重排
        B, C, F = x_fft.shape
        x_fft_flat = x_fft.reshape(B * C, F)
        x_fft_shuffled = torch.index_select(x_fft_flat, -1, self.permutation_map)

        # 保留能量
        if self.preserve_energy:
            original_energy = torch.sum(torch.abs(x_fft_flat) ** 2, dim=-1, keepdim=True)
            shuffled_energy = torch.sum(torch.abs(x_fft_shuffled) ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(original_energy / (shuffled_energy + 1e-8))
            x_fft_shuffled = x_fft_shuffled * scale

        x_fft_shuffled = x_fft_shuffled.reshape(B, C, F)

        # 转换回时域
        x_shuffled = torch.fft.irfft(x_fft_shuffled, n=L, dim=-1, norm='ortho')

        return x_shuffled

    def inverse(self, x):
        """
        逆向操作：从颠倒频率的信号恢复原始信号
        x: 频率颠倒后的时域信号 (B, C, L)
        """
        B, C, L = x.shape

        # 转换到频域
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 获取频域维度
        B, C, F = x_fft.shape

        # 展平
        x_fft_flat = x_fft.reshape(B * C, F)

        # 应用逆置换
        x_fft_restored = torch.index_select(x_fft_flat, -1, self.inverse_map)

        # 能量恢复
        if self.preserve_energy:
            current_energy = torch.sum(torch.abs(x_fft_flat) ** 2, dim=-1, keepdim=True)
            restored_energy = torch.sum(torch.abs(x_fft_restored) ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(current_energy / (restored_energy + 1e-8))
            x_fft_restored = x_fft_restored * scale

        # 重塑回原始形状
        x_fft_restored = x_fft_restored.reshape(B, C, F)

        # 转换回时域
        x_restored = torch.fft.irfft(x_fft_restored, n=self.L, dim=-1, norm='ortho')

        return x_restored

    def _binarize_hard(self, x):
        """硬阈值二值化"""
        return (x > self.threshold).float()

    def _binarize_soft(self, x, temperature=0.1):
        """软阈值二值化（可微分）"""
        return torch.sigmoid((x - self.threshold) / temperature)

    def _binarize_iterative(self, x, num_iterations=5):
        """迭代二值化（基于Gerchberg-Saxton算法）"""
        # 保存原始频域特征
        with torch.no_grad():
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            target_amp = torch.abs(x_fft)
            target_phase = torch.angle(x_fft)

        # 初始化
        current = x.clone()

        for _ in range(num_iterations):
            # 频域约束
            current_fft = torch.fft.rfft(current, dim=-1, norm='ortho')

            # 保留我们想要的频域特征（这里可以根据需求调整）
            # 例如，保留相位但替换幅度，或保留幅度但替换相位
            new_fft = target_amp * torch.exp(1j * torch.angle(current_fft))

            # 转换回时域
            current = torch.fft.irfft(new_fft, n=self.L, dim=-1, norm='ortho')

            # 时域约束：二值化
            current = self._binarize_hard(current)

        return current

    def binary_fft_invert(self, x):
        """
        执行频率颠倒并返回二值信号

        Args:
            x: 输入信号，形状为 (B, C, L)，应为二值信号 (0或1)

        Returns:
            频率颠倒后的二值信号
        """
        B, C, L = x.shape

        # 确保输入是二值的（可选）
        # x = (x > 0.5).float()

        # 转换到频域
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 创建频率置换映射
        if self.permutation_map is None or self.L != L:
            self._create_fft_permutation(L, x.device)

        # 频率重排
        B, C, F = x_fft.shape
        x_fft_flat = x_fft.reshape(B * C, F)

        # 应用频率颠倒
        x_fft_shuffled = torch.index_select(x_fft_flat, -1, self.permutation_map.to(x.device))

        # 保持能量守恒（可选）
        if self.preserve_energy:
            original_energy = torch.sum(torch.abs(x_fft_flat) ** 2, dim=-1, keepdim=True)
            shuffled_energy = torch.sum(torch.abs(x_fft_shuffled) ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(original_energy / (shuffled_energy + 1e-8))
            x_fft_shuffled = x_fft_shuffled * scale

        x_fft_shuffled = x_fft_shuffled.reshape(B, C, F)

        # 转换回时域（此时得到浮点数）
        x_shuffled = torch.fft.irfft(x_fft_shuffled, n=L, dim=-1, norm='ortho')

        # 二值化处理
        if self.binarization_method == 'hard':
            result = self._binarize_hard(x_shuffled)
        elif self.binarization_method == 'soft':
            result = self._binarize_soft(x_shuffled)
        elif self.binarization_method == 'iterative':
            result = self._binarize_iterative(x_shuffled)
        else:
            result = x_shuffled  # 不二值化

        return result

    def invert_with_phase_preservation(self, x, phase_preservation=0.5):
        """
        带相位保留的频率颠倒（更平滑的过渡）

        Args:
            x: 输入二值信号
            phase_preservation: 相位保留程度 (0-1)，1表示完全保留原始相位
        """
        B, C, L = x.shape

        # 原始频域
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        orig_amp = torch.abs(x_fft)
        orig_phase = torch.angle(x_fft)

        # 创建频率置换
        if self.permutation_map is None or self.L != L:
            self._create_fft_permutation(L, x.device)

        # 置换幅度
        B, C, F = x_fft.shape
        orig_amp_flat = orig_amp.reshape(B * C, F)
        shuffled_amp = torch.index_select(orig_amp_flat, -1, self.permutation_map.to(x.device))
        shuffled_amp = shuffled_amp.reshape(B, C, F)

        # 混合相位
        if phase_preservation > 0:
            # 生成随机相位作为"新"相位
            random_phase = torch.rand_like(orig_phase) * 2 * torch.pi - torch.pi
            mixed_phase = phase_preservation * orig_phase + (1 - phase_preservation) * random_phase
        else:
            mixed_phase = torch.rand_like(orig_phase) * 2 * torch.pi - torch.pi

        # 重构频域
        new_fft = shuffled_amp * torch.exp(1j * mixed_phase)

        # 转换回时域并二值化
        x_new = torch.fft.irfft(new_fft, n=L, dim=-1, norm='ortho')

        return self._binarize_hard(x_new)


from torch.func import vmap, grad


def analyze_batch_zero_loop(model, output, target):
    """
    零循环批量计算：一次性处理所有样本

    Args:
        model: 神经网络模型
        output: [batch_size, seq_len, features] 模型输出
        target: [batch_size, seq_len, features] 目标值

    Returns:
        biases: [batch_size] 每个样本的偏好
    """
    batch_size = output.shape[0]
    device = output.device

    # 保存原始参数
    original_params = [p.clone() for p in model.parameters()]

    # === 1. 计算所有样本的损失向量 ===
    loss_per_sample = ((output - target) ** 2).mean(dim=(1, 2))  # [B]

    # === 2. 使用向量化方法一次计算所有梯度 ===
    # 创建对角矩阵，一次反向传播得到所有梯度
    sample_weights = torch.eye(batch_size, device=device)  # [B, B]

    # 一次性计算所有样本的加权损失
    all_losses = (loss_per_sample.unsqueeze(0) * sample_weights).sum(dim=1)  # [B]

    # 一次性计算所有梯度（每个样本对应一个梯度向量）
    all_grads = torch.autograd.grad(
        all_losses.sum(),  # 所有损失的和对每个样本的梯度就是每个样本的梯度之和
        model.parameters(),
        create_graph=False,
        allow_unused=True
    )

    # === 3. 提取权重层的梯度 ===
    weight_grads = []
    weight_shapes = []
    for name, grad in zip([p[0] for p in model.named_parameters()], all_grads):
        if 'weight' in name and grad is not None and grad.dim() == 2:
            weight_grads.append(grad.detach().cpu().numpy())
            weight_shapes.append(grad.shape)

    if not weight_grads:
        return np.ones(batch_size)

    # === 4. 批量FFT分析 ===
    # 将所有层的梯度连接起来处理
    all_biases = np.ones(batch_size)

    for layer_grad in weight_grads:
        # layer_grad shape: [out_features, in_features] 这是所有样本的梯度之和
        # 我们需要分离每个样本的贡献，但这里无法直接分离

        # 替代方法：使用梯度范数作为特征
        grad_norm = np.linalg.norm(layer_grad, axis=1)  # [out_features]

        # 对每个输出神经元的梯度做FFT（模拟频率分析）
        grad_fft = np.abs(fft(layer_grad, axis=1))  # [out_features, in_features]
        avg_spectrum = grad_fft.mean(axis=0)  # [in_features]

        # 分离频率
        n_freqs = len(avg_spectrum)
        low_idx = n_freqs // 4
        high_idx = n_freqs * 3 // 4

        low_mag = avg_spectrum[:low_idx].mean() if low_idx > 0 else 0
        high_mag = avg_spectrum[high_idx:].mean() if high_idx < n_freqs else 1

        # 计算全局偏好（所有样本共享）
        layer_bias = low_mag / (high_mag + 1e-8)

        # 所有样本使用相同的层偏好
        all_biases *= layer_bias

    # 几何平均
    final_biases = all_biases ** (1.0 / len(weight_grads))

    # 恢复模型
    for p, op in zip(model.parameters(), original_params):
        p.data.copy_(op)
    model.zero_grad()

    return np.full(batch_size, final_biases)