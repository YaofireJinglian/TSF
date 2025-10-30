import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import einops
import torch.fft as fft
from torch.nn.modules.linear import Linear
import sys
sys.setrecursionlimit(3000) 

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model, seq_len, d_state, dconv, expand, dropout, decomp_kernel, ch_ind, tpm_num_scales, ssm_state_dim):
        super(Block, self).__init__()
        self.tpm = TPM(d_model=d_model, num_scales=tpm_num_scales, ssm_state_dim=ssm_state_dim)
        self.fdm = FDM(d_model=d_model, seq_len=seq_len, ssm_state_dim=ssm_state_dim)
        self.dfm = DFM(d_model=d_model, d_state=d_state, dconv=dconv, expand=expand, dropout=dropout, decomp_kernel=decomp_kernel, ch_ind=ch_ind)

    def forward(self, x):
        # x: [B, L, D]
        tpm_out = self.tpm(x)                     # [B, L, D]
        fdm_out = self.fdm(x)                     # [B, L, D]
        block_output = self.dfm(x, tpm_out, fdm_out)  # [B, L, D]
        return block_output

class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers 
        
        self.d_model = configs.n1 
        
        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.enc_in)
            
        self.proj_in = torch.nn.Linear(self.seq_len, self.d_model)
        
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(
                d_model=self.enc_in, 
                seq_len=self.d_model, 
                d_state=self.configs.d_state,
                dconv=self.configs.dconv,
                expand=self.configs.e_fact,
                dropout=self.configs.dropout,
                decomp_kernel=25, 
                ch_ind=self.configs.ch_ind,
                tpm_num_scales=3, 
                ssm_state_dim=self.configs.d_state 
            ))


        k_size = 7 
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(nn.Conv1d(
                in_channels=self.enc_in, 
                out_channels=self.enc_in, 
                kernel_size=k_size, 
                padding=(k_size - 1) // 2
            ))
        
        self.liner_conv = torch.nn.Linear(self.seq_len, self.d_model) #  L -> n1
        self.final_proj = torch.nn.Linear(2 * self.d_model, self.pred_len) #  2*n1 -> pred_len
        
    def forward(self, x):
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')

        x_block_perm = x.permute(0, 2, 1) # -> [B, D, L]
        x_block_proj = self.proj_in(x_block_perm) # -> [B, D, n1]
        x_block = x_block_proj.permute(0, 2, 1) # -> [B, n1, D]

        x_conv = x.permute(0, 2, 1) # -> [B, D, L]
        for i in range(self.num_layers):
            x_block = self.blocks[i](x_block) # [B, n1, D]
            x_conv = self.convs[i](x_conv)   # [B, D, L]
        out_block_proj = x_block.permute(0, 2, 1) # -> [B, D, n1]
        out_liner = self.liner_conv(x_conv) # [B, D, L] -> [B, D, n1]
        out_concat = torch.cat([out_block_proj, out_liner], dim=-1) # -> [B, D, 2*n1]
        output = self.final_proj(out_concat) # -> [B, D, pred_len]
        output = output.permute(0, 2, 1) # -> [B, pred_len, D]
        
        if self.configs.revin == 1:
            output = self.revin_layer(output, 'denorm')
            
        return output

class TPM(nn.Module):
    def __init__(self, in_dim, num_scales, scale_dims, ssm_hierarchical_dim, ssm_fine_grained_dim, ff_hidden_dim, dropout_p=0.1):
        super().__init__()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)

        self.proj_in_top = nn.Linear(in_dim, in_dim)
        self.causal_convs = nn.ModuleList([
            CausalConv1d(in_dim, scale_dims[i], kernel_size=2 * (i + 1) - 1)
            for i in range(num_scales)
        ])
        concat_dim = sum(scale_dims)
        self.proj_y = nn.Linear(concat_dim, ssm_hierarchical_dim)
        self.hierarchical_ssm = SSM(ssm_hierarchical_dim, ssm_hierarchical_dim)
        self.proj_out_top = nn.Linear(ssm_hierarchical_dim, in_dim)

        self.proj_in_bottom = nn.Linear(in_dim, in_dim)
        self.feed_forward = FeedForward(in_dim, ff_hidden_dim)
        self.proj_ff = nn.Linear(in_dim, ssm_fine_grained_dim)
        self.fine_grained_ssm = SSM(ssm_fine_grained_dim, ssm_fine_grained_dim)
        self.proj_out_bottom = nn.Linear(ssm_fine_grained_dim, in_dim)

    def forward(self, x):
        # x: [B, L, D]
        x_residual = x

        x_top = self.proj_in_top(x)               # [B, L, D]
        x_top_act = self.silu(x_top)              # [B, L, D]
        x_trans = x_top_act.transpose(1, 2)       # [B, D, L]
        scale_outputs = []
        for conv in self.causal_convs:
            conv_out = conv(x_trans)              # [B, D_out_i, L]
            scale_outputs.append(conv_out.transpose(1, 2))  # [B, L, D_out_i]
        y_fused = torch.cat(scale_outputs, dim=-1)           
        y_gating = self.proj_y(y_fused)                     # [B, L, H_hier]
        y_ssm = self.hierarchical_ssm(y_gating)              # [B, L, H_hier]
        hier_gated = y_gating * y_ssm                       # [B, L, H_hier]
        hier_out = self.proj_out_top(hier_gated)             # [B, L, D]

        x_bottom = self.dropout(x)
        x_bottom = self.proj_in_bottom(x_bottom)             # [B, L, D]
        x_bottom_act = self.silu(x_bottom)                   # [B, L, D]
        ff_out = self.feed_forward(x_bottom_act)             # [B, L, D]
        ff_gating = self.proj_ff(ff_out)                     # [B, L, H_fine]
        ff_ssm = self.fine_grained_ssm(ff_gating)            # [B, L, H_fine]
        fine_gated = ff_gating * ff_ssm                      # [B, L, H_fine]
        fine_out = self.proj_out_bottom(fine_gated)          # [B, L, D]

        output = x_residual + hier_out + fine_out            # [B, L, D]
        return output


class FDM(nn.Module):
    def __init__(self, k, seq_dim, feature_dim, fssm_hidden_dim, ff_hidden_dim):
        
        super().__init__()
        
        self.k = k

        self.proj_q = nn.Linear(feature_dim, feature_dim)
        self.proj_k = nn.Linear(feature_dim, feature_dim)
        self.proj_v = nn.Linear(feature_dim, feature_dim)
        self.ssm_q = SSM(feature_dim, feature_dim)
        self.ssm_k = SSM(feature_dim, feature_dim)
        self.ssm_v = SSM(feature_dim, feature_dim)
        self.silu = nn.SiLU()
        self.ff_v = FeedForward(feature_dim, fssm_hidden_dim) 
        self.series_decomp = SeriesDecomp(feature_dim)
        self.feed_forward = FeedForward(feature_dim, ff_hidden_dim) 
    
    def _fft(self, x):
        return fft.fft(x, dim=1) 
    
    def _ifft(self, x):
        return fft.ifft(x, dim=1).real

    def forward(self, x):
        
        Q = self.proj_q(x) 
        K = self.proj_k(x) 
        V = self.proj_v(x) 
        q_fft = self._fft(Q)
        q_ssm = self.ssm_q(q_fft) 
        k_fft = self._fft(K)
        k_ssm = self.ssm_k(k_fft) 
        v_act = self.silu(V)
        v_ssm = self.ssm_v(v_act)
        v_path = self.ff_v(v_ssm) 
        qk_prod = q_ssm * torch.conj(k_ssm) 
        qk_path = self._ifft(qk_prod)       
        fssm_output = qk_path * v_path 
        decomp_trend, _ = self.series_decomp(fssm_output)
        x_current = x 
        for _ in range(self.k):
            x_residual_iter = x_current
            ff_input = x_residual_iter + decomp_trend
            ff_output = self.feed_forward(ff_input)
            x_current = x_residual_iter + ff_output
        
        return x_current
class DFM(nn.Module):
    def __init__(self, d_model, d_state, dconv, expand, dropout, decomp_kernel, ch_ind=1):
        super(DFM, self).__init__()
        self.d_model = d_model
        self.ch_ind = ch_ind
        self.decompsition = series_decomp(decomp_kernel)
        self.dropout_layer = nn.Dropout(dropout)
        self.mamba_freq = Mamba(d_model=self.d_model, d_state=d_state, d_conv=dconv, expand=expand)
        self.mamba_time = Mamba(d_model=1 if ch_ind == 1 else self.d_model, d_state=d_state, d_conv=dconv, expand=expand)

    def forward(self, x, tpm_out, fdm_out):
        # x, tpm_out, fdm_out: [B, L, D]
        device = x.device
        x_fused = x + tpm_out + fdm_out                 # [B, L, D]
        x_perm = x_fused.permute(0, 2, 1)               # [B, D, L]
        B, D, L = x_perm.shape
        seasonal_init, trend_init = self.decompsition(x_perm)
        seasonal = self.dropout_layer(seasonal_init).unsqueeze(0)
        trend = self.dropout_layer(trend_init).unsqueeze(0)
        seasonal_fft = torch.fft.rfft(seasonal.contiguous())
        trend_fft = torch.fft.rfft(trend.contiguous())
        st = seasonal_fft * torch.conj(trend_fft)
        st = torch.fft.irfft(st, n=L).squeeze(0)        # [B, D, L]
        x_freq_out = self.mamba_freq(st.to(device))     # [B, D, L]
        x_time_in = self.dropout_layer(x_perm)          # [B, D, L]
        if self.ch_ind == 1:
            x_time_in_reshaped = x_time_in.reshape(B * D, 1, L)
            x_time_out_reshaped = self.mamba_time(x_time_in_reshaped.to(device))
            x_time_out = x_time_out_reshaped.reshape(B, D, L)   # [B, D, L]
        else:
            x_time_out = self.mamba_time(x_time_in.to(device))  # [B, D, L]
        dfm_out_perm = x_freq_out + x_time_out          # [B, D, L]
        dfm_out = dfm_out_perm.permute(0, 2, 1)         # [B, L, D]
        return dfm_out
class SSM(nn.Module):

    def __init__(self, d_model, state_dim=64, bidirectional=False):

        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        

        self.A_real = nn.Parameter(torch.randn(d_model, state_dim))
        self.A_imag = nn.Parameter(torch.randn(d_model, state_dim))
        self.B = nn.Parameter(torch.randn(d_model, state_dim))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))
        self.D = nn.Parameter(torch.randn(d_model))
        

        self._init_parameters()
        

        if bidirectional:
            self.A_real_rev = nn.Parameter(torch.randn(d_model, state_dim))
            self.A_imag_rev = nn.Parameter(torch.randn(d_model, state_dim))
            self.B_rev = nn.Parameter(torch.randn(d_model, state_dim))
            self.C_rev = nn.Parameter(torch.randn(d_model, state_dim))
            self._init_parameters_rev()
    
    def _init_parameters(self):

        nn.init.normal_(self.A_real, mean=0.0, std=0.2)
        nn.init.normal_(self.A_imag, mean=0.0, std=0.2)
        

        nn.init.xavier_normal_(self.B)
        nn.init.xavier_normal_(self.C)
        

        nn.init.constant_(self.D, 0.1)
    
    def _init_parameters_rev(self):

        nn.init.normal_(self.A_real_rev, mean=0.0, std=0.2)
        nn.init.normal_(self.A_imag_rev, mean=0.0, std=0.2)
        nn.init.xavier_normal_(self.B_rev)
        nn.init.xavier_normal_(self.C_rev)
    
    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        A = self._compute_discrete_A()
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            u_t = x[:, t, :]  # [batch_size, d_model]
            state = torch.einsum('bd, bsd -> bs', u_t, self.B.unsqueeze(0)) + \
                    torch.einsum('bs, bsd -> bd', state, A)
            y_t = torch.einsum('bs, bsd -> bd', state, self.C.unsqueeze(0)) + \
                  torch.einsum('bd, d -> bd', u_t, self.D)
            outputs.append(y_t.unsqueeze(1))
        output = torch.cat(outputs, dim=1)  # [batch_size, seq_len, d_model]

        if self.bidirectional:
            A_rev = self._compute_discrete_A(reverse=True)
            state_rev = torch.zeros(batch_size, self.state_dim, device=x.device)
            outputs_rev = []
            for t in reversed(range(seq_len)):
                u_t = x[:, t, :]
                state_rev = torch.einsum('bd, bsd -> bs', u_t, self.B_rev.unsqueeze(0)) + \
                           torch.einsum('bs, bsd -> bd', state_rev, A_rev)
                y_t_rev = torch.einsum('bs, bsd -> bd', state_rev, self.C_rev.unsqueeze(0)) + \
                         torch.einsum('bd, d -> bd', u_t, self.D)
                outputs_rev.insert(0, y_t_rev.unsqueeze(1))
            
            output_rev = torch.cat(outputs_rev, dim=1)
            output = output + output_rev  
        
        return output
    
    def _compute_discrete_A(self, reverse=False):
        if reverse and self.bidirectional:
            real = self.A_real_rev
            imag = self.A_imag_rev
        else:
            real = self.A_real
            imag = self.A_imag
        real = F.tanh(real) * 0.5
        A_complex = torch.complex(real, imag)
        I = torch.eye(self.state_dim, device=real.device).unsqueeze(0)
        A_plus = I + 0.5 * A_complex.unsqueeze(1)
        A_minus = I - 0.5 * A_complex.unsqueeze(1)
        A_inv = torch.inverse(A_plus)
        A_d_complex = torch.bmm(A_inv, A_minus)
        A_d = A_d_complex.real
        return A_d
        
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
  

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
    
        x = self.linear1(x)
        
        x = F.relu(x)
        
        x = self.dropout(x)
      
        x = self.linear2(x)
        return x

class Bottleneck_Construct(nn.Module):
    
    def __init__(self, d_model, window_size, d_inner, device='cuda:0'):
        super(Bottleneck_Construct, self).__init__()
        self.device = torch.device(device)  

        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size, self.device),
                ConvLayer(d_inner, window_size, self.device),
                ConvLayer(d_inner, window_size, self.device)
            ]).to(self.device)  
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i], self.device))
            self.conv_layers = nn.ModuleList(self.conv_layers).to(self.device)

        self.up = nn.Linear(d_inner, d_model).to(self.device)
        self.down = nn.Linear(d_model, d_inner).to(self.device)
        self.norm = nn.LayerNorm(d_model).to(self.device)

    def forward(self, enc_input):
        enc_input = enc_input.to(self.device)  # Move input to the correct device
        seq_len = enc_input.size()[1]

        temp_input = self.down(enc_input).permute(0, 2, 1)  # CSCM Linear entrance layer
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2).to(self.device)  # Concatenation
        all_inputs = self.up(all_inputs).to(self.device)  # CSCM Linear exit layer
        all_inputs = torch.cat([enc_input, all_inputs], dim=1).to(self.device)  # Concatenation with original input
        all_inputs = all_inputs.permute(0, 2, 1).to(self.device)

        liner1 = nn.Linear(all_inputs.size()[2], seq_len).to(self.device)  # Linear layer on the correct device
        all_inputs = liner1(all_inputs).to(self.device)
        all_inputs = all_inputs.permute(0, 2, 1).to(self.device)

        all_inputs = self.norm(all_inputs)

        return all_inputs


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size, device='cuda:0'):  # Added device parameter
        super(ConvLayer, self).__init__()
        self.device = torch.device(device)  # Store the device
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size).to(self.device)  # Ensure the layer is on the correct device
        self.norm = nn.BatchNorm1d(c_in).to(self.device)
        self.activation = nn.ELU().to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 1dim 

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):


    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean





