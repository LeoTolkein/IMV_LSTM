from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class IMVTensorLSTM(torch.jit.ScriptModule):
    """
    IMV Tensor LSTM + Mixture Attention

    注：这个结构自带一个 output linear layer，形状由 output_dim 和 n_units 决定。

    - 输入：
        - input_dim: input dimension
        - output_dim: output dimension
        - n_units: num of hidden states
        - init_std: Gaussian initialization 时的标准差
    """

    # * 在 PyTorch 的 TorchScript（即 `torch.jit.ScriptModule`）中，
    # * `__constants__` 是一个特殊的类属性，用于指定哪些成员变量在模型被
    # * “脚本化”（script）时应当被视为常量。
    # * 优化与序列化：这样做的好处是，TorchScript 可以对这些常量做更多优化，
    # * 并且在模型序列化（保存/加载）时，这些值会被直接存储在模型文件里。
    # * 区别于普通属性：如果你没有把某个属性加到 `__constants__`，
    # * TorchScript 会把它当作普通的 Python 属性处理，
    # * 可能会导致脚本化时出错，或者属性值在模型保存/加载后丢失。
    # * - 何时需要用？
    # * -- 你希望某些参数在模型脚本化后保持不变（如网络结构超参数）。
    # * -- 这些参数不是 `nn.Parameter`（即不会被训练），只是模型结构的描述。

    # * 此处将 n_units 和 input_dim 视为常量
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        # * 此处使用的是正态分布初始化
        # * ---------- LSTM 部分 ---------- 
        # * input_dim 对应原文中 N 的维度
        # * weight matrices for inputs x to cell state update j
        # * j 是这篇论文使用的特殊符号，基本等同于 standard LSTM 中的 c_tilde
        # * 原文中 $U_j \in \mathbb{R}^{N \times d \times d_0}$，但是这里
        # * 我们假设每一个输入和输出都是标量，所以 $d_0 = 1$
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        # * weight matrices for inputs x to input gate i
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        # * weight matrices for inputs x to forget gate f
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        # * weight matrices for inputs x to output gate o
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        # * weight matrices for previous hidden states h to cell state update j
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        # * weight matrices for previous hidden states h to input gate i
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        # * weight matrices for previous hidden states h to forget gate f
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        # * weight matrices for previous hidden states h to output gate o
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        # * biases for states (广义的) update
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        # * ---------- Mixture attention 部分 ----------
        # * F_alpha_n, F_alpha_n_b, 对应文章中的时序注意力部分
        # * 时序注意力计算需要特殊的 einsum 操作，因此不能使用 nn.Linear，
        # * 需要自己手动实现。
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.Phi = nn.Linear(2*n_units, output_dim)
        # * F_beta 对应文章中的变量注意力部分，这一部分相对 trivial 所以用
        # * nn.Linear 实现。
        self.F_beta = nn.Linear(2*n_units, 1)
        self.n_units = n_units
        self.input_dim = input_dim
    

    @torch.jit.script_method
    def forward(self, x):
        # * hidden state 和 cell state 的初始化
        # * 此处的 .cuda() 操作在正规的代码里不推荐，因为要求必须有支持 
        # * cuda 的GPU，如果没有会直接扔报错。
        # * h_tilda_t \in (batch_size, input_dim, n_units)
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        # * c_tilda_t \in (batch_size, input_dim, n_units)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # * eq 1 -- cell state candidate
            # * j_tilda_t \in (batch_size, input_dim, n_units)
            j_tilda_t = torch.tanh(
                # * b = batch_size, i = input_dim, j = n_units, k = n_units
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j)
                # * x[:,t:t+1,:] 的形状为 (batch_size, 1, input_dim)
                # * b = batch_size, i = 1, j = input_dim, k = n_units
                # * 此处同 W 的写法有区别主要是因为 x 和 h 的形状不同
                + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_j) 
                # * b_j 与前两项相加，遵循 PyTorch 默认的
                # * 广播规则：首先，从右向左匹配维度。基于这一规则，
                # * b_j 会先被unsqueeze 到 (1, input_dim, n_units)
                # * 然后再在 dim0 上广播 batch_size 次，最后
                # * 与 outputs 相加。
                + self.b_j
            )
            # * eq 5: input gate, forget gate, output gate
            # * i_tilda_t \in (batch_size, input_dim, n_units)
            i_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i)
                + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_i)
                + self.b_i
            )
            # * f_tilda_t \in (batch_size, input_dim, n_units)
            f_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f)
                + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_f)
                + self.b_f
            )
            # * o_tilda_t \in (batch_size, input_dim, n_units)
            o_tilda_t = torch.sigmoid(
                torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o)
                + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_o)
                + self.b_o
            )
            # * eq 6: cell state update
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # * eq 7: hidden state update
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]  # * 保存每个时间步的 hidden state
        # * 将 outputs 堆叠成一个张量，形状为 
        # * (seq_len, batch_size, input_dim, n_units)
        # * 因为 stack 默认 dim = 0, 所以 第 0 维 是 seq_len
        outputs = torch.stack(outputs)
        # * 将 outputs 的维度变换为 (batch_size, seq_len, input_dim, n_units)
        outputs = outputs.permute(1, 0, 2, 3)
        # * eq 8 ~ eq 13，mixture attention 与最终预测结果计算
        # * ------- 时序注意力 ---------
        # * 1. 计算 f^n ( h^n_t ) 项，这一部分本质就是一个使用 tanh 作为激活
        # * 函数的 FCN，只是因为 input 有3个维度，与 PyTorch 默认的 Linear
        # * 层形状不匹配，所以需要自己实现。
        alphas = torch.tanh(
            # * outputs: (batch_size, seq_len, input_dim, n_units)
            # * F_alpha_n: (input_dim, n_units, 1)
            # * 在 dim3，即 n_units 对应的维度上相乘并累加。
            # * F_alpha_n 对应维度 k 的形状为1 （对应 alpha 在乘 h 的时候，
            # * 视为标量，这和 d0 = 1 的假设没有关系）。因此乘完后的形状为
            # * (batch_size, seq_len, input_dim, 1)
            torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) 
            # * 然后将 F_alpha_n_b 加到 F_alpha_n 上，相加遵循 PyTorch 默认的
            # * 广播规则：首先，从右向左匹配维度。基于这一规则，
            # * F_alpha_n_b 会先被unsqueeze 到 (1, 1, input_dim, 1)
            # * 然后再在 dim0 和 dim1 上广播 batch_size 和 seq_len 次，最后
            # * 与 outputs 相加。
            + self.F_alpha_n_b
        )
        # * 此时 alphas 的形状为 (batch_size, seq_len, input_dim, 1)
        # * 2. 使用 softmax 计算得到 \alpha^n_t. 计算时 softmax 保留alphas 的
        # * 形状，只是在 dim=1 (即 seq_len 时间步维度) 上进行 softmax 计算
        alphas = F.softmax(alphas, dim=1)
        # * alphas 的最终形状为 (batch_size, seq_len, input_dim, 1)
        # * 3. 计算 $g^n = \sum_t \alpha^n_t h^n_t$，相乘发生在 dim3, 即
        # * n_units 对应的维度上，因此相乘后形状依然为 (batch_size, seq_len,
        # * input_dim, n_units)。随后在 dim1 上求和，得到的 g_n 的形状为
        # * (batch_size, input_dim, n_units)
        g_n = torch.sum(alphas*outputs, dim=1)
        # * 4. 将 $g^n$ 和 $h^n_T$ 拼接起来，此处 h_tilda_t 即为 hidden state
        # * 序列的最后一个时间步的 hidden state，形状为 
        # * (batch_size, input_dim, n_units)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        # * hg 的形状为 (batch_size, input_dim, 2*n_units)
        # * 5. 计算 $\mu_n = \phi_n(h^n_T \oplus g^n)$. 此处并未计算整个分布
        # * 只计算了 $\mu_n$。Phi 会沿着 dim1 即 input_dim 分别计算每个变量的
        # * 最终注意力值期望 mu_n.
        mu = self.Phi(hg)
        # * mu 的形状为 (batch_size, input_dim, output_dim)
        # * ------- 变量注意力 ---------
        # * 这部分对应文章中的 $\Pr(z_{T+1} = n | h^1_T \oplus g^1, \dots,
        # * h^N_T \oplus g^N)$，使用 softmax 计算每个变量的注意力分数。
        betas = torch.tanh(self.F_beta(hg))
        betas = F.softmax(betas, dim=1)
        # * betas 的最终形状为 (batch_size, input_dim, 1)
        # * ------- 最终预测值 ---------
        # * 计算最终的预测值：将 betas 和 mu 在 dim1 上相乘并求和。
        mean = torch.sum(betas*mu, dim=1)
        # * mean 的形状为 (batch_size, output_dim)

        # * ---------- 如何得到 I 和 T^N? ---------
        # * 1. 计算 I
        # * I = 1/M * sum(q_m) along m axis. 即在 sample size 维度上对 q_m 求和，
        # * 然后除以 sample size M. q^n_m = mu_n * beta^n. 

        # * 2. 计算 T^N
        # * T^N = 1/M * sum(alpha^n_m) along m axis, 即在 sample size 维度上对
        # * alpha^n_m 求和，然后除以 sample size M.
        # * alpha^n_m 为 [alpha^n_1, alpha^n_2, ..., alpha^n_T] 的向量。

        return mean, alphas, betas


class IMVFullLSTM(torch.jit.ScriptModule):
    """
    IMV Full LSTM
    """

    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        # * 每个 W 的输入均为 [input_dim + input_dim*n_units]，其中
        # * 第1个 input_dim 对应 x_t，后续 input_dim*n_units 对应
        # * h_tilda_t.view() 即展平后的 hidden state。
        # * 输出为了与 hidden state 的形状相匹配，因此是 input_dim*n_units
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim


    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # * eq 1
            j_tilda_t = torch.tanh(
                    torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) 
                    + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_j) 
                    + self.b_j)
            inp = torch.cat(
                # * .view() 对应论文中的 vec() 操作
                [x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)],dim=1)
            # * eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # * eq 3, .view() 对应论文中的 vec() 操作
            c_t = c_t*f_t + i_t*j_tilda_t.view(j_tilda_t.shape[0], -1)
            # * eq 4, .view() 对应论文中的 matricization() 操作
            h_tilda_t = (o_t*torch.tanh(c_t)).view(
                h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # * eq 8 ~ eq 13，mixture attention 与最终预测结果计算
        alphas = torch.tanh(
            torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) 
            + self.F_alpha_n_b
        )
        alphas = F.softmax(alphas, dim=1)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = F.softmax(betas, dim=1)
        mean = torch.sum(betas*mu, dim=1)

        return mean, alphas, betas