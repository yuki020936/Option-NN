import matplotlib
# matplotlib.use('Agg') # TODO: uncomment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid, simpson
from utils import feature_scale, inverse_feature_scale
import torch
import json
import numpy as np
import pandas as pd
import config
import os
import sys
matplotlib.use('Agg')  
def plot_risk_neutral_density(pdf_dataset, model, result_folder: str, test_date: str, input_fscale_dict: dict, output_fscale_dict: dict, daily_plot_off, is_residual, is_residual_itm):
    # TODO: plot risk neutral density for residual model. remember to split itm part and otm part
    # estimate the risk neutral probability distribution for an asset price at a future time from the volatility surface
    # refer to Hull J.C. - Options, Futures, and Other Derivatives p. 447 Addendix, DETERMINING IMPLIED RISK-NEUTRAL DISTRIBUTIONS FROM VOLATILITY SMILES
    # f(K) = e^{r\tau}*(c1+c3-2*c2)/delta^2, where c1, c2, c3 are call option with strike price K-delta, K, K+delta respectively
    # https://quantoisseur.com/category/python/
    #pdf_dataset都是同個date同個tau
    if daily_plot_off:
        return
    eps = 0.5
    market_volumes = pdf_dataset['volume'].values
    # print(f"volumes = {market_volumes}")
    if len(market_volumes) < 10:
        return
    tau_str =  int(pdf_dataset['tau'][0]*252)
    print(f'plot_risk_neutral_density, date = {test_date}, tau = {tau_str}')
    result_df_path = f"{result_folder}/raw_Q_density_{test_date}{'_itm' if is_residual and is_residual_itm else '_otm' if is_residual and not is_residual_itm else ''}_tau{tau_str}_current.csv"
    result_image_path = f"{result_folder}/raw_Q_density_{test_date}_tau{tau_str}_current.png"
    opposite_residual_result_path = f"{result_folder}/raw_Q_density_{test_date}{'_itm' if is_residual and not is_residual_itm else '_otm' if is_residual and is_residual_itm else ''}_tau{tau_str}_current.csv"
    if not os.path.exists(result_df_path):
        
        market_Ks = (pdf_dataset['strike_price']).values
        market_ImVols = (pdf_dataset['impl_volatility']).values
        # 1. 排序
        sorted_indices = np.argsort(market_Ks)
        market_Ks = np.array(market_Ks)[sorted_indices]
        market_ImVols = np.array(market_ImVols)[sorted_indices]

        # 2. 移除重複值，確保 market_Ks 嚴格遞增
        unique_indices = np.diff(market_Ks) > 0
        market_Ks = market_Ks[np.insert(unique_indices, 0, True)]
        market_ImVols = market_ImVols[np.insert(unique_indices, 0, True)]
        market_tau = pdf_dataset['tau'][0]
        market_r = pdf_dataset['r'][0]
        market_d = pdf_dataset['d'][0]
        market_S = pdf_dataset['S'][0]
        index_put_call_strike_price_equal_to_underlying_price = (market_Ks == market_S)
        if sum(market_Ks == market_S) > 1:
            # since we only convert out of money put option into in the money call option, at the money (strike price == underlying asset price) option will have both types of put and call
            # market_Ks will not be strictly increasing
            return

        fit_func = InterpolatedUnivariateSpline(market_Ks, market_ImVols, k=4)  # order = 4
        simulate_Ks = np.arange(max(0, market_Ks[0] - 100), market_Ks[-1] + 100)
        simulate_ImVols = fit_func(simulate_Ks)
        filter = (simulate_ImVols > 0)
        if is_residual & is_residual_itm:
            filter = filter & (simulate_Ks < market_S * 1.1)
        elif is_residual & (not is_residual_itm):
            filter = filter & (simulate_Ks > market_S * 0.9)
        simulate_ImVols = simulate_ImVols[filter]
        simulate_Ks = simulate_Ks[filter]

        # TODO: RuntimeWarning: invalid value encountered in log
        # TODO: RuntimeWarning: divide by zero encountered in double_scalars
        BS = lambda S, K, tau, d, r, sigma: S * np.exp(-d * tau) * norm.cdf(
            ((np.log(S / K) + ((r - d) + 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))).item()) - K * np.exp(
            -r * tau) * norm.cdf(
            ((np.log(S / K) + ((r - d) - 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))).item())

        # start to plot
        probs_bs, probs_model = [], []
        for K, vol in zip(simulate_Ks, simulate_ImVols):
            # ground truth model
            c1_bs = BS(market_S, K - eps, market_tau, market_d, market_r, vol)
            c2_bs = BS(market_S, K, market_tau, market_d, market_r, vol)
            c3_bs = BS(market_S, K + eps, market_tau, market_d, market_r, vol)
            prob_bs = np.exp(market_r * market_tau) * (c1_bs + c3_bs - 2 * c2_bs) / (eps ** 2)
            probs_bs.append(prob_bs.item())

            # testing model
            invm, vol = torch.tensor(K / market_S).to(model.device), torch.tensor(vol).to(model.device)
            tau, S, r, d, = torch.tensor(market_tau).to(model.device), torch.tensor(market_S).to(
                model.device), torch.tensor(market_r).to(model.device), torch.tensor(market_d).to(model.device)
            # Z-score feature scaling accord. train dataset before feeding into model & return scaled output to unscaled output & return to call price by multiplying S*exp(-r*tau)
            scaled_c1_model = model(feature_scale(x=(invm * S - eps) / S, mean=input_fscale_dict['invm_mean'],
                                                  std=input_fscale_dict['invm_std']),
                                    feature_scale(x=tau, mean=input_fscale_dict['tau_mean'],
                                                  std=input_fscale_dict['tau_std']))  # K = invm*S
            scaled_c2_model = model(
                feature_scale(x=invm, mean=input_fscale_dict['invm_mean'], std=input_fscale_dict['invm_std']),
                feature_scale(x=tau, mean=input_fscale_dict['tau_mean'], std=input_fscale_dict['tau_std']))
            scaled_c3_model = model(feature_scale(x=(invm * S + eps) / S, mean=input_fscale_dict['invm_mean'],
                                                  std=input_fscale_dict['invm_std']),
                                    feature_scale(x=tau, mean=input_fscale_dict['tau_mean'],
                                                  std=input_fscale_dict['tau_std']))
            c1_model = inverse_feature_scale(y=scaled_c1_model, mean=output_fscale_dict['mean'],
                                             std=output_fscale_dict['std'],
                                             shift=output_fscale_dict['shift']) * S * torch.exp(
                -r * tau)  # model output is c/S
            c2_model = inverse_feature_scale(y=scaled_c2_model, mean=output_fscale_dict['mean'],
                                             std=output_fscale_dict['std'],
                                             shift=output_fscale_dict['shift']) * S * torch.exp(-r * tau)
            c3_model = inverse_feature_scale(y=scaled_c3_model, mean=output_fscale_dict['mean'],
                                             std=output_fscale_dict['std'],
                                             shift=output_fscale_dict['shift']) * S * torch.exp(-r * tau)
            prob_model = torch.exp(r * tau) * (c1_model + c3_model - 2 * c2_model) / (eps ** 2)
            probs_model.append(prob_model.item())

        dist_dict = {'KorS': list(simulate_Ks), 'prob_bs': probs_bs, 'prob_model': probs_model,'m':list(market_S/simulate_Ks), 'iv': list(simulate_ImVols),'tau':[market_tau]*len(list(simulate_Ks))}
        
        dist_df = pd.DataFrame.from_dict(dist_dict)
        print(f'is_residual:{is_residual}, is_residual_itm:{is_residual_itm}')
        dist_df = dist_df[dist_df['KorS'] < market_S] if is_residual and is_residual_itm else dist_df[dist_df['KorS'] > market_S] if is_residual and (not is_residual_itm) else dist_df
        dist_df.to_csv(result_df_path, index=False)
    else:
        dist_df = pd.read_csv(result_df_path)
        simulate_Ks = np.array(dist_df['KorS'])
        probs_bs = np.array(dist_df['prob_bs'])
        probs_model = np.array(dist_df['prob_model'])
        simulate_ImVols = np.array(dist_df['iv'])

    if not os.path.exists(result_image_path) and (not is_residual or (is_residual and os.path.exists(result_df_path) and os.path.exists(opposite_residual_result_path))):
        if is_residual:
            opposite_dist_df = pd.read_csv(opposite_residual_result_path)
            whole_dist_df = pd.concat([dist_df, opposite_dist_df], ignore_index=True).sort_values(by=['KorS'])
            simulate_Ks = np.array(whole_dist_df['KorS'])
            probs_bs = np.array(whole_dist_df['prob_bs'])
            probs_model = np.array(whole_dist_df['prob_model'])
            simulate_ImVols = np.array(whole_dist_df['iv'])

        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.suptitle(f"implied distribution, {test_date}, tau{tau_str}")
        fig.text(0.5, 0.04, 'S_T or K', ha='center')
        fig.text(0.03, 0.65, 'prob. of S&P 500 index = S_T at time T', va='center', rotation='vertical')

        axs[0].set_title(f"ground truth model's integral={sum(probs_bs):.3f}")
        axs[0].plot(simulate_Ks, probs_bs)

        axs[1].set_title(f"testing model's integral={sum(probs_model):.3f}")
        axs[1].plot(simulate_Ks, probs_model)

        axs[2].set_title(f"implied volatility curve")
        axs[2].set_ylabel("implied volatilities")
        axs[2].plot(simulate_Ks, simulate_ImVols)

        plt.savefig(result_image_path)
        #plt.show()
        plt.close()
    return dist_df

def plot_syn_risk_neutral_density(pdf_dataset, model, result_folder: str, test_date: str, input_fscale_dict: dict, output_fscale_dict: dict, daily_plot_off, is_residual, is_residual_itm):
    #for syn option price
    if daily_plot_off:
        return
    
    dx = np.diff(pdf_dataset['strike_price'])

    tau_str =  int(pdf_dataset['tau'][0]*252)
    print(f'plot_syn_risk_neutral_density, date = {test_date}, tau = {tau_str}')
    whole_residual_result_path = f"{result_folder}/Q_density_files"
    os.makedirs(whole_residual_result_path, exist_ok=True)
    result_df_path = f"{result_folder}/raw_Q_density_{test_date}{'_itm' if is_residual and is_residual_itm else '_otm' if is_residual and not is_residual_itm else ''}_tau{tau_str}_syn.csv"
    result_image_path = f"{result_folder}/raw_Q_density_{test_date}_tau{tau_str}_syn.png"
    opposite_residual_result_path = f"{result_folder}/raw_Q_density_{test_date}{'_itm' if is_residual and not is_residual_itm else '_otm' if is_residual and is_residual_itm else ''}_tau{tau_str}_syn.csv"
    whole_residual_result_path = f"{whole_residual_result_path}/raw_Q_density_{test_date}_tau{tau_str}_syn.csv"
    if not os.path.exists(result_df_path):
        
        K = (pdf_dataset['strike_price']).values
        ImVols = (pdf_dataset['iv']).values
        call_pred = (pdf_dataset['call_pred']).values
        
        valid_mask = ~np.isnan(ImVols)
        K = K[valid_mask]
        ImVols = ImVols[valid_mask]
        call_pred = call_pred[valid_mask]
        # 1. 排序
        sorted_indices = np.argsort(call_pred)
        K = np.array(K)[sorted_indices]
        ImVols = np.array(ImVols)[sorted_indices]
        call_pred = np.array(call_pred)[sorted_indices]

        # 2. 移除重複值，確保 market_Ks 嚴格遞增
        unique_indices = np.diff(call_pred) > 0
        call_pred = call_pred[np.insert(unique_indices, 0, True)]
        K = K[np.insert(unique_indices, 0, True)]
        ImVols = ImVols[np.insert(unique_indices, 0, True)]
        
        tau =  pdf_dataset['tau'][0]
        r = pdf_dataset['r'][0]
        d = pdf_dataset['d'][0]
        S = pdf_dataset['S'][0]
        
        index_put_call_strike_price_equal_to_underlying_price = (K == S)
        if sum(K == S) > 1:
            # since we only convert out of money put option into in the money call option, at the money (strike price == underlying asset price) option will have both types of put and call
            # market_Ks will not be strictly increasing
            return
        probs_model = []
        idx_list = []
        for idx in range(1,len(call_pred)-1):
            idx_list.append(idx)
            c1_model = call_pred[idx-1]
            c2_model = call_pred[idx]
            c3_model = call_pred[idx+1]
            prob_model = np.exp(r * tau) * (c1_model + c3_model - 2 * c2_model) / (dx[idx] ** 2)
            if np.isfinite(prob_model.item()) and prob_model.item() >= np.float64(-1e-6):  # 允許小的負值（數值誤差）
                prob_model = max(np.float64(0.0), prob_model)
            probs_model.append(prob_model.item())
        probs_model = np.array(probs_model)
        idx_list = np.array(idx_list)
        mask = probs_model >= np.float64(0.0)
        probs_model = probs_model[mask]
        idx_list = idx_list[mask]
        K = K[idx_list]
        #probs_model = gaussian_filter1d(probs_model, sigma=0.5) #平滑
        #對K的pdf = g(K)，轉換成對數報酬率log(K/S)的pdf = g(K)Se^log(K/S)
        rt = np.log(K/S)
        exp_rt =np.exp(rt)
        probs_log_rt = probs_model*S*exp_rt
    
        dist_dict = {'KorS': list(K),'prob_model': probs_model,'m':list(S/K),'invm':list(K/S),'log_rt':rt,'prob_log_rt':probs_log_rt, 'iv': list(ImVols[idx_list]),'tau':[tau]*len(list(K))}
        
        dist_df = pd.DataFrame.from_dict(dist_dict)
        dist_df = dist_df[dist_df['KorS'] < S] if is_residual and is_residual_itm else dist_df[dist_df['KorS'] > S] if is_residual and (not is_residual_itm) else dist_df
        dist_df.to_csv(result_df_path, index=False)
    else:
        dist_df = pd.read_csv(result_df_path)
        simulate_Ks = np.array(dist_df['KorS'])
        probs_model = np.array(dist_df['prob_model'])
        simulate_invm = np.array(dist_df['invm'])
        simulate_log_rt = np.array(dist_df['log_rt'])
        probs_log_rt = np.array(dist_df['prob_log_rt'])
        simulate_ImVols = np.array(dist_df['iv'])

    if not os.path.exists(result_image_path) and (not is_residual or (is_residual and os.path.exists(result_df_path) and os.path.exists(opposite_residual_result_path))):
        if is_residual:
            opposite_dist_df = pd.read_csv(opposite_residual_result_path)
            
            whole_dist_df = pd.concat([dist_df, opposite_dist_df], ignore_index=True).sort_values(by=['KorS']).reset_index(drop=True)
            simulate_Ks = np.array(whole_dist_df['KorS'])
            probs_model = np.array(whole_dist_df['prob_model'])
            simulate_ImVols = np.array(whole_dist_df['iv'])
            CDF_KorS = cal_CDF_trapez(simulate_Ks,probs_model)
            whole_dist_df['CDF_KorS'] = CDF_KorS
            
            simulate_invm = np.array(whole_dist_df['invm'])
            simulate_log_rt = np.array(whole_dist_df['log_rt'])
            probs_log_rt = np.array(whole_dist_df['prob_log_rt'])
            CDF_log_rt = cal_CDF_trapez(simulate_log_rt,probs_log_rt)
            whole_dist_df['CDF_log_rt'] = CDF_log_rt
            whole_dist_df.to_csv(whole_residual_result_path, index=False)

        fig, axs = plt.subplots(3, 1)
        
        fig.suptitle(f"{test_date}, tau {tau_str}, S0 {pdf_dataset['S'][0]:.3f}")
        fig.text(0.5, 0, 'S_T or K', ha='center')
        #fig.text(0.03, 0.65, 'prob. of S&P 500 index = S_T at time T', va='center', rotation='vertical')

        axs[0].set_title(f"synthetic option implied distribution, CDF={np.max(CDF_KorS):.3f}")
        axs[0].set_ylabel(f"prob. of S&P 500 index ")
        axs[0].plot(simulate_Ks, probs_model)
        
        axs[1].set_title(f"synthetic option implied distribution, CDF={np.max(CDF_log_rt):.3f}")
        axs[1].set_ylabel(f"prob. of S&P 500 index log return ")
        axs[1].plot(simulate_log_rt, probs_log_rt)
        axs[1].set_xlim(-1, simulate_log_rt[-1])
        
        axs[2].set_title(f"implied volatility curve")
        axs[2].set_ylabel("implied volatilities")
        mask = simulate_ImVols <= np.float64(1.2)
        axs[2].plot(simulate_Ks[mask], simulate_ImVols[mask])
        axs[2].axhline(y=1, color='#9370DB', linestyle='--', label='IV=1')
        axs[2].legend()
        plt.tight_layout()
        plt.savefig(result_image_path)
        #plt.show()
        plt.close()
        return whole_dist_df
    
    elif os.path.exists(whole_residual_result_path):
        whole_dist_df = pd.read_csv(whole_residual_result_path)
        return whole_dist_df
    return dist_df

def cal_CDF_trapez (x,pdf):
    cdf = np.concatenate([[0], np.cumsum(np.diff(x) * (pdf[:-1] + pdf[1:]) / 2)])
    return cdf         

def plot_date_loss_curve(loss_dict: dict, result_folder: str, test_date: str, is_merge=False, daily_plot_off=False):
    # list average loss for each epoch
    loss_df = pd.DataFrame.from_dict(loss_dict)
    loss_df.index.names = ['epoch']
    loss_df.to_csv(f"{result_folder}/loss_message_{test_date}.csv", index=True)
    if daily_plot_off:
        return
    if is_merge:
        plt.title('merging loss curve\n(to see more info please check itm_part and otm_part folder)')
    else:
        plt.title('loss curve')
        # TODO: make sure that this argmin val loss' criteria is consistent with train.py's argmin val loss' criteria
        plt.axvline(x=np.argmin(np.array(loss_dict['val_mse_list']) + np.array(loss_dict['val_mape_list'])), ymin=0, ymax=1, color='black', linestyle='--', label='min_val_metric')
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    plt.plot(np.log(loss_dict['train_loss_list']), label='train_loss', color='green', linestyle='-', alpha=1)
    plt.plot(np.log(loss_dict['train_mse_list']), label='train_mse', color='brown', linestyle='-', alpha=1)
    plt.plot(np.log(loss_dict['train_mape_list']), label='train_mape', color='blue', linestyle='-', alpha=1)
    plt.plot(np.log(loss_dict['val_loss_list']), label='val_loss', color='green', linestyle='--', alpha=0.7)
    plt.plot(np.log(loss_dict['val_mse_list']), label='val_mse', color='brown', linestyle='--', alpha=0.7)
    plt.plot(np.log(loss_dict['val_mape_list']), label='val_mape', color='blue', linestyle='--', alpha=0.7)
    plt.plot(np.log(loss_dict['test_loss_list']), label='test_loss', color='green', linestyle=':', alpha=0.4)
    plt.plot(np.log(loss_dict['test_mse_list']), label='test_mse', color='brown', linestyle=':', alpha=0.4)
    plt.plot(np.log(loss_dict['test_mape_list']), label='test_mape', color='blue', linestyle=':', alpha=0.4)
    loss_curve_moving_average = [np.mean(loss_dict['val_loss_list'][max(0, i-config.moving_average_epoch):i+1]) for i in range(len(loss_dict['val_loss_list']))]
    plt.plot(np.log(loss_curve_moving_average), label= 'val_loss(ma)', color='black', linestyle='-', alpha = 0.4)
    plt.legend(loc='best')
    plt.savefig(f"{result_folder}/loss_{test_date}.png")
    #plt.show()
    #plt.clf()
    plt.close()

def plot_model_avg_loss(result_folder: str, model_name: str, model_loss_list_dict: dict):
    # list average loss for each date
    # with open(f"{result_folder}/avg_loss_of_model_{model_name}_among_different_dates.json", "w") as fin:
        # fin = sys.stdout
        # json.dump(model_loss_list_dict, fin)
        # fin.seek(0)
        # data = json.load(fin)
    model_loss = pd.DataFrame().from_dict(model_loss_list_dict)
    model_loss.rename(columns={'date_list': 'date'}, inplace=True)
    model_loss.to_csv(f"{result_folder}/avg_loss_of_model_{model_name}_among_different_dates.csv", index=False)

def plot_learning_rate(result_folder: str, lr_list: list, daily_plot_off):
    if daily_plot_off:
        return
    plt.xlabel("iteration")
    plt.ylabel("learning rate")
    plt.plot(lr_list)
    plt.savefig(f"{result_folder}/learning_rate.png")
    #plt.show()
    plt.close()

def plot_lr_range_test(lr_list, loss_list, max_grad_idx):
    plt.plot(lr_list, loss_list, color='black', linewidth=1)
    plt.scatter(lr_list, loss_list, color='black', alpha=0.3)
    plt.scatter(lr_list[max_grad_idx], loss_list[max_grad_idx],
                label=f"steepest gradient with LR={lr_list[max_grad_idx]:.3f}", marker='s', edgecolors='red', facecolors='red', s=60)
    # plt.scatter(lr_list[stable_grad_idx], loss_list[stable_grad_idx],
    #             label=f"stable gradient with LR={lr_list[stable_grad_idx]:.3f}", marker='^', edgecolors='red', facecolors='red', s=60)
    plt.xscale("log")
    plt.title("learning rate range test")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.legend()
    #plt.show()
    plt.close()

def plot_relation_between_option_and_stirke(result_folder: str, date: str, exdate: str, call: list, K: list, volume: list, is_syn: list, S: float):
    title = f"qtdate_{date}_exdate_{exdate}"
    plt.title(title)
    plt.xlabel("strike price")
    plt.ylabel("call option price")
    plt.axvline(x=S, ymin=0, ymax=1, color='black', linestyle='--', label='underlying asset price')
    # plt.plot(K[~is_syn], call[~is_syn], color='black', alpha=0.8)
    plt.plot(K, call, color='black', alpha=0.8)
    rgba_colors = np.zeros((len(call), 4))
    rgba_colors[:, 0] = is_syn
    rgba_colors[:, 3] = volume / volume.sum()
    rgba_colors[:, 3] += 1 - rgba_colors[:, 3].max()
    rgba_colors[:, 3] -= min(0.2, rgba_colors[:, 3].min())
    plt.scatter(K[~is_syn], call[~is_syn], color=rgba_colors[~is_syn], label='market data')
    if True in is_syn:
        plt.scatter(K[is_syn], call[is_syn], color=rgba_colors[is_syn], label='synthetic data')
    plt.legend()
    plt.savefig(os.path.join(result_folder, title + ".png"))
    plt.close()

def plot_3d_relation(x, y, z_dict: dict, xlabel, ylabel, zlabel, title, angle=(30, 330), path=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(30, 300)
    ax.view_init(*angle) # 30, 330
    # ax.view_init(30, 360)
    # ax.view_init(135, 45)
    for z_label, z_value in z_dict.items():
        ax.scatter(x, y, z_value, s=20, label=z_label)
    ax.plot_trisurf(x, y, list(z_dict.values())[0], alpha=0.2, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.invert_xaxis()
    ax.set_title(title)
    plt.legend()
    if path is not None:
        plt.savefig(path)
    #plt.show()
    #plt.clf()
    plt.close()

def check_risk_neutral_density_valid():
    # TODO: plot the relation between strike price (x-axis) and call option price (y-axis) to see whether it is concave up
    # TODO: on eval date, x-axis=K/S, y-axis=tau, z-axis=call option price
    pass
