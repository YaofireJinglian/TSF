import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import einops
from torch.nn.modules.linear import Linear
import sys
sys.setrecursionlimit(3000) 

class Block(torch.nn.Module): # Block
    def __init__(self,d_state,dconv,expand,n1,n2,d_model_param2,seq_len,pred_len,dropout,decomp_kernel,ch_ind,l):
        super(Block, self).__init__()

        
        self.d_state = d_state
        self.dconv = dconv
        self.expand = expand
        self.n1 = n1
        self.n2 = n2
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.decomp_kernel = decomp_kernel
        self.ch_ind = ch_ind
        self.l = l
        self.decompsition = series_decomp(self.decomp_kernel)
        self.lin1=torch.nn.Linear(self.seq_len,self.n1)
        self.dropout=torch.nn.Dropout(self.dropout)
        self.lin2 = torch.nn.Linear(self.n1,self.n2)
        self.d_model_param1 = 1 # 128
        self.d_model_param2 = d_model_param2
        self.mamba1 = Mamba(d_model=self.d_model_param1,d_state=self.d_state,d_conv=self.dconv,expand=self.expand) 
        self.mamba2 = Mamba(d_model=self.d_model_param2,d_state=self.d_state,d_conv=self.dconv,expand=self.expand) 
    def forward(self,x):
        device = x.device
        x = torch.permute(x, (0, 2, 1))
        if self.ch_ind == 1:
            x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2])) 
        if self.l == 1:
           
            x = self.lin1(x.to(device).to(self.lin1.weight.dtype))
        if self.l == 2:
            x = self.lin1(x.to(device).to(self.lin1.weight.dtype))
            x = self.lin2(x.to(device).to(self.lin1.weight.dtype))
           
        x_res1 = x
        seasonal_init, trend_init = self.decompsition(x)
        seasonal, trend = self.dropout(seasonal_init).unsqueeze(0), self.dropout(trend_init).unsqueeze(0)
        seasonal = torch.fft.rfft(seasonal.contiguous(),)
        trend = torch.fft.rfft(trend.contiguous(),)
        st = seasonal * torch.conj(trend)
        st = torch.fft.irfft(st, n=seasonal_init.shape[2], ).squeeze(0)
        x1 = self.mamba2(st.to(device)) 
       
        if self.ch_ind == 1:
            
            x2 = torch.permute(x, (0, 2, 1))
            x2 = self.dropout(x2)
        else:
            x2 = x
       
        x2 = self.mamba1(x2.to(device))  # 1792,512,1
       
        if self.ch_ind == 1:
            x2 = torch.permute(x2, (0, 2, 1))
        x2 = x2 + x1 + x_res1 # 1792,1,512
        # print(x2.shape)
        return x2


class Model(torch.nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.revin==1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.b1 = Block(d_state=self.configs.d_state,dconv=self.configs.dconv,expand=self.configs.e_fact,
        n1=self.configs.n1,n2=self.configs.n2,d_model_param2=self.configs.n1,seq_len=self.configs.seq_len,
        pred_len=self.configs.pred_len,dropout=self.configs.dropout,decomp_kernel=25,ch_ind=self.configs.ch_ind,l=1)

        self.b2 = Block(d_state=self.configs.d_state,dconv=self.configs.dconv,expand=self.configs.e_fact,
        n1=self.configs.n1,n2=self.configs.n2,d_model_param2=self.configs.n2,seq_len=self.configs.seq_len,
        pred_len=self.configs.pred_len,dropout=self.configs.dropout,decomp_kernel=25,ch_ind=self.configs.ch_ind,l=2)
        

        self.lin3=torch.nn.Linear(self.configs.n2,self.configs.n1)
        self.lin4=torch.nn.Linear(2*self.configs.n1,self.configs.pred_len)
        self.ln = nn.LayerNorm(configs.n1)
        self.ln2 = nn.LayerNorm(self.configs.n2)

        self.fd1 = FeedForward(configs.n1,2048)
        self.fd2 = FeedForward(self.configs.n2,2048)

        

    def forward(self, x):
       
         device = x.device 
         x=self.revin_layer(x,'norm')
         
    
         x1 = self.b1(x)
         # FFD--------------------------------
         x1 = self.ln(x1) + x1
         x1_fd1 = self.fd1(x1)
         x1 = self.ln(x1_fd1+x1)
         # FFD--------------------------------
         x2 = self.b2(x)
         # FFD -------------------------------
         x2 = self.ln2(x2) + x2
         x2_fd2 = self.fd2(x2)
         x2 = self.ln2(x2_fd2+x2)
         # FFD -------------------------------
         x2 = self.lin3(x2.to(device))  # 1792,1,512
        #  x3 = self.b1(x)
        #  x4 = self.b2(x)
         x = torch.cat([x2, x1], dim=2)  # 1792,1,1024
         x = self.lin4(x.to(device))  # 1792,1,96
         if self.configs.ch_ind == 1:
            x = torch.reshape(x, (-1, self.configs.enc_in, self.configs.pred_len))  # 256,7,96
         x = torch.permute(x, (0, 2, 1))  # 256,96,7
         if self.configs.revin==1:
             x=self.revin_layer(x,'denorm')
         else:
             x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
             x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
         return x


class SSM(nn.Module):

    def __init__(self, d_model, state_dim=64, bidirectional=False):
        """
        d_model: 输入/输出维度
        state_dim: 状态空间维度（隐藏状态大小）
        bidirectional: 是否双向
        """
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





