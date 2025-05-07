from typing import List
import torch
from torch import nn, Tensor


class IMVTensorLSTM(torch.jit.ScriptModule):
    """
    IMV Tensor LSTM

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
        # * input_dim 对应原文中 N 的维度
        # * weight matrices for inputs x to cell state update j
        # * j 是这篇论文使用的特殊符号，基本等同于 standard LSTM 中的 c_tilde
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
        # * weight matrices for hidden states h to alpha
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        # * weight matrices for hidden states h to beta
        self.F_beta = nn.Linear(2*n_units, 1)
        # * weight matrices for hidden states h to output
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    
    @torch.jit.script_method
    def forward(self, x):
        # * hidden state 
        # * 猜测： x.shape[0] 是 batch size
        # * 后两个维度分别代表 input_dim 和 hidden_dim
        # * 此处的 .cuda() 操作在正规的代码里不推荐，因为要求必须有支持 
        # * cuda 的GPU，如果没有会直接扔报错。
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1 -- cell state update
            j_tilda_t = torch.tanh(
                        torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j)
                        + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_j) 
                        + self.b_j
                    )
            # eq 5
            i_tilda_t = torch.sigmoid(
                        torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i)
                        + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_i)
                        + self.b_i
                    )
            f_tilda_t = torch.sigmoid(
                        torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f)
                        + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_f)
                        + self.b_f
                    )
            o_tilda_t = torch.sigmoid(
                        torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o)
                        + torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_o)
                        + self.b_o
                    )
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
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
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t:t+1,:], self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.view(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        return mean, alphas, betas