import torch
import torch.nn as nn

# 建構Activation Function
# Squareplus => f(x, b) = 0.5 * (x + root(x^2 + b))
# why need slope? => func. 中的0.5可視為1*0.5，把它改成某個constant c*0.5
class Squareplus(nn.Module):
    def __init__(self, b=100, slope=1):
        # 沒有繼承為何需要用super? => 繼承了nn.Module
        super(Squareplus, self).__init__()
        '''
        # nn.parameter : 將張量標記為模型參數，模型參數為訓練過程中需要優化的可學習參數
        # nn.init.normal_ : 對張量進行 normal dist. 的初始化，mean 為 0、std 為 0.01
        # torch.empty(1, 1) : 建一個 1*1 的空 tensor，類似於一個 1*1 的 matrix
        '''
        # choose self.b by backward propagation
        # self.b = nn.Parameter(nn.init.normal_(torch.empty(1, 1, dtype=torch.float64), mean=0, std=0.01))
        # self.slope = nn.Parameter(nn.init.normal_(torch.empty(1, 1, dtype=torch.float64), mean=0, std=0.01))
        # self.softplus = nn.Softplus()
        
        # 用 backward propagation 的方式 train 出來的結果很糟，故不採用
                
        # choose self.b by hand
        self.b = b
        self.slope = slope

    def forward(self, x):
        # choose self.b and self.slope by backward propagation
        # x = (self.slope*self.slope) * 0.5 * (x + torch.sqrt(x * x + 0.1 + (self.b*self.b)))
        # 公式為何不同? => 一次方和二次方都有嘗試，此處僅留下二次方
        
        # choose self.b and self.slope by hand
        x = self.slope * 0.5 * (x + torch.sqrt(x*x+self.b))
        return x

# 建構Activation Function
# Softplus => f(x, b) = log(1 + exp(x))
# why need b? => func. 中的x可視為1*x，把它改成某個constant c*x
class Softplus(nn.Module):
    def __init__(self, beta=1):
        super(Softplus, self).__init__()
        self.beta = beta

    def forward(self, x):
        # 公式為何不同? => look "why need b?"
        x = torch.log(1 + torch.exp(self.beta * x)) / self.beta
        return x
    
# Single model = sum^J(sigma1(invm_bia - invm*exp(invm_weight)) * sigma2(tau_bia - tau*exp(tau_weight)) * exp(pricing_weight))
# sigma1 = Softplus(AAAI) or Squareplus(學姊)
# sigma2 = Sigmoid
class SingleModel(nn.Module):
    # residual_model_call_itm_part : 打開的話是學姊的論文，沒打開的話是AAAI
    # 以下是在進行Single model的參數設定
    # device='cpu' : 將tensor儲存在cpu上
    def __init__(self, J=5, device='cpu', residual_model_call_itm_part=False):
        super(SingleModel, self).__init__()
        self.device = device
        self.residual_model_call_itm_part = residual_model_call_itm_part
        self.invm_bias =        nn.Parameter(nn.init.normal_(torch.empty(J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.invm_weights =     nn.Parameter(nn.init.normal_(torch.empty(J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.tau_bias =         nn.Parameter(nn.init.normal_(torch.empty(J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.tau_weights =      nn.Parameter(nn.init.normal_(torch.empty(J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.pricing_weights =  nn.Parameter(nn.init.normal_(torch.empty(J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.sigma1 = Squareplus() if residual_model_call_itm_part else nn.Softplus()
        self.sigma2 = nn.Sigmoid()

    # Activation Function & first_diif 矩陣運算的邏輯?
    def forward(self, invm, tau): # (invm, tau) : (inverse moneyness, time to maturity)
        # TODO : bug multi not monotonic with invm !?
        # element-wise computation : 逐元素計算，對相同維度的集合中的每個元素進行相同的運算(非矩陣乘法)
        if self.residual_model_call_itm_part:
            first_term = self.sigma1(self.invm_bias + invm*torch.exp(self.invm_weights)) # element-wise computation
        else:
            first_term = self.sigma1(self.invm_bias - invm * torch.exp(self.invm_weights))  # element-wise computation
        second_term = self.sigma2(self.tau_bias + tau*torch.exp(self.tau_weights)) # element-wise computation
        # torch.sum(..., axis=0) : 把column中的每個元素相加
        output = torch.sum(first_term * second_term * torch.exp(self.pricing_weights), axis=0) # element-wise computation
        return output

    def first_diff_wrt_invm(self, invm, tau):
        # invm.view(1, -1) : 把invm改成 1*n 的tensor
        # invm.view(-1, 1) : 把invm改成 n*1 的tensor
        # A @ B : @為PyTorch中處理矩陣乘法的符號
        return torch.sum(-torch.exp(self.invm_weights) * self.sigma2(self.invm_bias - torch.exp(self.invm_weights)@invm.view(1,-1)) * self.sigma2(self.tau_bias + torch.exp(self.tau_weights)@tau.view(1,-1)) * torch.exp(self.pricing_weights), axis=0)

class MultiModel(nn.Module):
    # self._get_name()
    
    # residual_model_call_itm_part : 打開的話是學姊的論文，沒打開的話是AAAI
    # 以下是在進行Multi model的參數設定
    # device='cpu' : 將tensor儲存在cpu上
    def __init__(self, J=5, I=9, K=5, device='cpu', residual_model_call_itm_part=False):
        super(MultiModel, self).__init__()
        self.device = device
        self.J = J # the number of hidden layer neurons in single models
        self.I = I # the number of single models in Multi
        self.K = K # the number of neurons in hidden layer for the right-branch weighting network of Multi
        self.sigmoid = nn.Sigmoid()  # sigma_2
        self.in_hid_weights = nn.Parameter(nn.init.normal_(torch.empty(self.K, 2, dtype=torch.float64), mean=0, std=0.01))
        self.in_hid_bias = nn.Parameter(nn.init.normal_(torch.empty(self.K, 1, dtype=torch.float64), mean=0, std=0.01))
        self.hid_out_weights = nn.Parameter(nn.init.normal_(torch.empty(self.K, self.I, dtype=torch.float64), mean=0, std=0.01))
        self.hid_out_bias = nn.Parameter(nn.init.normal_(torch.empty(self.I, dtype=torch.float64), mean=0, std=0.01))
        self.single_model_list = nn.ModuleList()
        for i in range(self.I):
            self.single_model_list.append(SingleModel(self.J, self.device, residual_model_call_itm_part))

    def forward(self, invm, tau):
        weight_numerator = self.sigmoid((self.in_hid_weights @ torch.vstack((invm, tau))) + self.in_hid_bias)  # shape=(K, 1)
        weight_denominator = torch.exp(weight_numerator.T @ self.hid_out_weights + self.hid_out_bias)  # shape = (1, I)
        weight_denominator = torch.sum(weight_denominator, axis=1) # shape = scalar
        output = 0
        for i in range(self.I):
            y_i = self.single_model_list[i](invm, tau) # single model's output
            weight_numerator_i = torch.sum(weight_numerator * self.hid_out_weights[:, i].view(-1, 1), axis=0) + self.hid_out_bias[i] #shape = scalar
            weight_numerator_i = torch.exp(weight_numerator_i)
            weight_i = weight_numerator_i / weight_denominator
            output += y_i*weight_i
        return output

    def first_diff_wrt_invm(self, invm, tau):
        # invm.requires_grad = True # to check correctness
        out_sigma2 = self.sigmoid((self.in_hid_weights @ torch.vstack((invm, tau))) + self.in_hid_bias)  # K * batchsize
        exp_term = torch.exp(out_sigma2.T @ self.hid_out_weights + self.hid_out_bias) # batchsize * I # torch.sum( ... , axis=1) # exp_term[:, i]
        d_exp_term_d_invm = ((out_sigma2 * (1 - out_sigma2) * self.in_hid_weights[:, 0].view(-1, 1)).T @ self.hid_out_weights) * exp_term # (K*batchsize).T @ (K*I) * (1*I) = batchsize * I # torch.sum( ... , axis=1) # exp_term[:, i]
        output = torch.zeros_like(invm)
        for i in range(self.I):
            d_w_i_d_invm = (torch.sum(exp_term, axis=1) * d_exp_term_d_invm[:, i] - exp_term[:, i]*torch.sum(d_exp_term_d_invm, axis=1)) / (torch.sum(exp_term, axis=1)**2)
            d_y_i_d_invm = self.single_model_list[i].first_diff_wrt_invm(invm, tau)
            y_i = self.single_model_list[i](invm, tau)
            w_i = exp_term[:, i] / torch.sum(exp_term, axis=1)
            output += d_y_i_d_invm*w_i + y_i*d_w_i_d_invm

        # check correctness
        # yy = self.forward(invm, tau)
        # yy.backward(torch.ones_like(invm))
        # output, invm.grad

        return output
