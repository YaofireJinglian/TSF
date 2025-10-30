from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ITFM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import pandas as pd
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ITFM':ITFM
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)        
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model:
                            outputs = self.model(batch_x)
                        
                else:
                    if 'Machine' in self.args.model:
                        outputs = self.model(batch_x)
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
        preds = np.array(preds)
        trues = np.array(trues)
        # print(preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe,_,_ = metric(preds, trues)
        
        print('mse:{}, mae:{},mape:{},rmae:{}'.format(mse, mae,mape,rmse))   
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
      
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                # print("第{i}",batch_x)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model:
                            outputs = self.model(batch_x)
                        

                        f_dim = -1 if self.args.features == 'MS' else 0
                        
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Machine' in self.args.model:
                            
                            outputs = self.model(batch_x)  # batch_x.shape = 16,96,7
                            # print(outputs.shape)
#--------------------------------------------------------------------------------------------------------------

                    # lamda = 0.5
                    # ax = 1-lamda
                    # rl = lamda
                    # rec_lambda = rl
                    # auxi_lambda = ax 
#--------------------------------------------------------------------------------------------------------------

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                   
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
#--------------------------------------------------------------------------------------------------------------

                    # loss = 0
                    # loss_rec = criterion(outputs, batch_y)
                    # loss += rec_lambda * loss_rec

                    # loss_auxi = compute_auxiliary_loss(outputs, batch_y, auxi_mode='rfft', auxi_type='phase', leg_degree=2,
                    #         mask=0.25, auxi_loss='MAE', module_first=1, device='cuda:0')
                    # loss += auxi_lambda * loss_auxi
#--------------------------------------------------------------------------------------------------------------

                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print('metric: 1st row is vail_set,2rd is test_set')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model:
                            outputs = self.model(batch_x)
                        
                else:
                    if 'Machine' in self.args.model:
                            outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    prd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, prd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model,x_shape=(batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        print('mse:{}, mae:{},mape:{},rmae:{}'.format(mse, mae,mape,rmse))   
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        temp_df = pd.DataFrame()
        temp_df['Seed']=[self.args.random_seed]
        temp_df['Model']=[self.args.model]
        temp_df['seq_len']=[self.args.seq_len]
        temp_df['label_len']=[self.args.label_len]
        temp_df['pred_len']=[self.args.pred_len]
        temp_df['n1']=[self.args.n1]
        temp_df['n2']=[self.args.n2]
        temp_df['dropout']=[self.args.dropout]
        temp_df['train_epochs']=[self.args.train_epochs]
        temp_df['batch']=[self.args.batch_size]
        temp_df['patience']=[self.args.patience]
        temp_df['LR']=[self.args.learning_rate]
        temp_df['dropout']=[self.args.dropout]
        temp_df['ch_ind']=[self.args.ch_ind]
        temp_df['revin']=[self.args.revin]
        temp_df['e_fact']=[self.args.e_fact]
        temp_df['dconv']=[self.args.dconv]

        temp_df['MSE']=[mse]
        temp_df['MAE']=[mae]
        temp_df['residual']=[self.args.residual]
        temp_df['d_state']=[self.args.d_state]

        temp_df['checkpoint_path']=[setting]

        if not os.path.exists('./csv_results/'+'result_'+self.args.data_path):
            temp_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)
        else:
            result_df=pd.read_csv('./csv_results/'+'result_'+self.args.data_path)
            result_df = pd.concat([result_df,temp_df],ignore_index=True)
            result_df.to_csv('./csv_results/'+'result_'+self.args.data_path, index=False)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Machine' in self.args.model:
                            outputs = self.model(batch_x)
                       
                else:
                    if 'Machine' in self.args.model:
                        outputs = self.model(batch_x)
                    
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
def compute_kl_loss(p, q, pad_mask=None):




    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
def compute_auxiliary_loss(outputs, batch_y, auxi_mode='mag', auxi_type=None, leg_degree=None,
                            mask=None, auxi_loss='MSE', module_first=False, device='cuda:0'):
    """
    计算辅助损失

    参数:
    - outputs (torch.Tensor): 模型输出
    - batch_y (torch.Tensor): 真实标签
    - auxi_mode (str): 辅助损失计算模式 ('fft', 'rfft', 'rfft-D', 'rfft-2D', 'legendre', 'chebyshev', 'hermite', 'laguerre')
    - auxi_type (str, optional): 对于 'rfft' 模式的类型 ('complex', 'complex-phase', 'complex-mag-phase', 'phase', 'mag', 'mag-phase')
    - leg_degree (int, optional): 用于多项式的度数（对于 'legendre', 'chebyshev', 'hermite', 'laguerre' 模式）
    - mask (torch.Tensor, optional): 计算损失时的掩码
    - auxi_loss (str): 辅助损失类型 ('MAE' 或 'MSE')
    - module_first (bool): 是否先计算辅助损失的绝对值
    - device (torch.device): 计算设备（如CPU或GPU）

    返回:
    - loss_auxi (torch.Tensor): 计算得到的辅助损失
    """
    loss_auxi = None

    if auxi_mode == "fft":
        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)

    elif auxi_mode == "rfft":
        if auxi_type == 'complex':
            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        elif auxi_type == 'complex-phase':
            loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
        elif auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
            loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif auxi_type == 'phase':
            loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
        elif auxi_type == 'mag':
            loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
        elif auxi_type == 'mag-phase':
            loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
            loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

    elif auxi_mode == "rfft-D":
        loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)

    elif auxi_mode == "rfft-2D":
        loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)

    elif auxi_mode == "legendre":
        if leg_degree is not None:
            loss_auxi = leg_torch(outputs, leg_degree, device=device) - leg_torch(batch_y, leg_degree, device=device)
        else:
            raise ValueError("leg_degree must be provided for 'legendre' mode.")

    elif auxi_mode == "chebyshev":
        if leg_degree is not None:
            loss_auxi = chebyshev_torch(outputs, leg_degree, device=device) - chebyshev_torch(batch_y, leg_degree, device=device)
        else:
            raise ValueError("leg_degree must be provided for 'chebyshev' mode.")

    elif auxi_mode == "hermite":
        if leg_degree is not None:
            loss_auxi = hermite_torch(outputs, leg_degree, device=device) - hermite_torch(batch_y, leg_degree, device=device)
        else:
            raise ValueError("leg_degree must be provided for 'hermite' mode.")

    elif auxi_mode == "laguerre":
        if leg_degree is not None:
            loss_auxi = laguerre_torch(outputs, leg_degree, device=device) - laguerre_torch(batch_y, leg_degree, device=device)
        else:
            raise ValueError("leg_degree must be provided for 'laguerre' mode.")
    else:
        raise NotImplementedError

    if mask is not None:
        loss_auxi *= mask

    if auxi_loss == "MAE":
        # MAE, 最小化element-wise error的模长
        loss_auxi = loss_auxi.abs().mean() if module_first else loss_auxi.mean().abs()
    elif auxi_loss == "MSE":
        # MSE, 最小化element-wise error的模长
        loss_auxi = (loss_auxi.abs()**2).mean() if module_first else (loss_auxi**2).mean().abs()
    else:
        raise NotImplementedError

    return loss_auxi

# rec_lambda = lambda

import torch
from numpy.polynomial import Chebyshev as C
from numpy.polynomial import Hermite as H
from numpy.polynomial import Laguerre as La
from numpy.polynomial import Legendre as L


def standard_laguerre(data, degree):
    tvals = np.linspace(0, 5, len(data))
    coeffs = La.fit(tvals, data, degree).coef

    laguerre_poly = La(coeffs)
    reconstructed_data = laguerre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def laguerre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(0, 5, T)
    laguerre_polys = np.array([La.basis(i)(tvals) for i in range(degree)])

    laguerre_polys = torch.from_numpy(
        laguerre_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals))
    coeffs_candidate = torch.mm(laguerre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(laguerre_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, laguerre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_hermite(data, degree):
    tvals = np.linspace(-5, 5, len(data))
    coeffs = H.fit(tvals, data, degree).coef

    hermite_poly = H(coeffs)
    reconstructed_data = hermite_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(degree)])

    hermite_polys = torch.from_numpy(
        hermite_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals ** 2))
    coeffs_candidate = torch.mm(hermite_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(hermite_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, hermite_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_leg(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = L.fit(tvals, data, degree).coef

    legendre_poly = L(coeffs)
    reconstructed_data = legendre_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)  # The Legendre series are defined in t\in[-1, 1]
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])  # Generate the basis functions which are sampled at tvals.
    # tvals = torch.from_numpy(tvals).to(device)
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    # This is implemented for 1D series.
    # For N-D series, here, the data matrix should be transformed as B,T,D -> B,D,T -> BD, T.
    # The legendre polys should be T,degree
    # Then, the dot should be a matrix multiplication: (BD, T) * (T, degree) -> BD, degree, which is the result of legendre transform.
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def standard_chebyshev(data, degree):
    tvals = np.linspace(-1, 1, len(data))
    coeffs = C.fit(tvals, data, degree).coef

    chebyshev_poly = C(coeffs)
    reconstructed_data = chebyshev_poly(tvals)
    return coeffs, reconstructed_data.reshape(-1)


def chebyshev_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)
    chebyshev_polys = np.array([C.basis(i)(tvals) for i in range(degree)])

    chebyshev_polys = torch.from_numpy(chebyshev_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(1 / torch.sqrt(1 - tvals ** 2))
    coeffs_candidate = torch.mm(chebyshev_polys, data) / torch.pi / T * 2
    # coeffs_candidate = torch.mm(torch.mm(chebyshev_polys, scale), data) / torch.pi * 2
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(chebyshev_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, chebyshev_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs