import torch
import torch.nn as nn
import math

from snns.surrogate_gradient import ArchAct, SpikeAct_extended


class GLIF(nn.Module):
    '''
    gated spiking neuron
    '''

    def __init__(self, inplace=2048):
        super(GLIF, self).__init__()
        # T 是内部仿真步长，与输入数据的维度无关
        self.T = 4
        self.soft_mode = False
        self.static_gate = False
        self.static_param = False
        self.time_wise = True
        self.plane = inplace
        tau, Vth, conduct = 0.25, 0.5, 0.5
        linear_decay = Vth / (self.T * 2)
        self.param = [tau, Vth, linear_decay, conduct]

        # c
        self.alpha, self.beta, self.gamma = [
            nn.Parameter(- math.log(1 / ((i - 0.5) * 0.5 + 0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
            for i in [0.6, 0.8, 0.6]]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                                         for i in self.param[:-1]]
        self.reVth = nn.Parameter(- math.log(1 / self.param[1] - 1) * torch.ones(self.plane, dtype=torch.float))
        # t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                        for i in self.param[3:]][0]

    def forward(self, x):  # 期望输入 x 的形状: (b, c, w), 例如 (32, 2048, 11)
        if x.shape[1] != self.plane:
            raise ValueError(f"输入通道维度错误! 模型期望 {self.plane}, 但输入为 {x.shape[1]}")

        # 初始化膜电位 u 和上一时刻的脉冲 o_prev
        # 它们的形状与输入 x 相同: (b, c, w)
        u = torch.zeros_like(x)
        o_prev = torch.zeros_like(x)

        # 用于存储每个时间步输出的脉冲序列
        out_sequence = []

        # 循环 T 次, 模拟神经元动态
        for step in range(self.T):
            # 在每个内部时间步，都使用相同的外部输入 x
            u, o_prev = self.extended_state_update(u, o_prev, x,
                                                  tau=self.tau.sigmoid(),
                                                  Vth=self.Vth.sigmoid(),
                                                  leak=self.leak.sigmoid(),
                                                  conduct=self.conduct[step].sigmoid(),
                                                  reVth=self.reVth.sigmoid())

            # out_sequence.append(o_new)
            # o_prev = o_new  # 更新上一时刻的脉冲状态

        # 将脉冲序列在新的时间维度 (dim=0) 上堆叠
        # 返回形状为 (T, b, c, w), 例如 (4, 32, 2048, 11)
        return o_prev

    # 状态更新函数，处理 (b, c, w) 形状的张量
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # 参数的形状为 (c,)，需要调整为 (1, c, 1) 以便与 (b, c, w) 的状态进行广播
        view_shape = (1, -1, 1)

        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(view_shape).clone().detach().sigmoid(), self.beta.view(
                    view_shape).clone().detach().sigmoid(), self.gamma.view(view_shape).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(view_shape).clone().detach().gt(0.).float(), self.beta.view(
                    view_shape).clone().detach().gt(0.).float(), self.gamma.view(view_shape).clone().detach().gt(
                    0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(view_shape).sigmoid(), self.beta.view(
                    view_shape).sigmoid(), self.gamma.view(view_shape).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(view_shape).sigmoid()), ArchAct.apply(
                    self.beta.view(view_shape).sigmoid()), ArchAct.apply(self.gamma.view(view_shape).sigmoid())

        # W_mul_o_t_n1 是恒定的输入电流 x
        # conduct 在每个时间步都不同，其形状为 (c,), 同样需要调整
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct.view(view_shape)))

        # 调整其他参数的形状
        tau_r = tau.view(view_shape)
        leak_r = leak.view(view_shape)
        reVth_r = reVth.view(view_shape)
        Vth_r = Vth.view(view_shape)

        # 更新膜电位
        u_t1_n1 = ((1 - al * (1 - tau_r)) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak_r) + \
                  I_t1 - (1 - ga) * reVth_r * o_t_n1.clone()

        # 根据膜电位是否超过阈值，产生脉冲
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth_r)
        return u_t1_n1, o_t1_n1

    def _initialize_params(self):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / self.param[0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / self.param[1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / self.param[1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / self.param[2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / self.param[3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))