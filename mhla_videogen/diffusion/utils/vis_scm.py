import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffusion.model.respace import compute_density_for_timestep_sampling


def visualize_scm_over_timesteps(
    model, pretrained_model, timesteps, clean_images, model_kwargs, sigma_data, save_dir, step=None
):
    """
    对给定的时间步长列表计算JVP并可视化结果

    Args:
        model: 模型
        timesteps (torch.Tensor): 要测试的时间步长列表，形状为[num_timesteps]
        x_t (torch.Tensor): 输入数据，形状为[batch_size, channels, height, width]
        model_kwargs (dict): 模型的额外参数
        sigma_data (float): sigma_data参数
        save_dir (str): 保存可视化结果的目录
        step (int, optional): 当前训练步数，用于文件命名
    """
    os.makedirs(save_dir, exist_ok=True)
    device = clean_images.device

    # 存储不同时间步的结果
    all_F_theta_grad = []
    all_F_theta_minus = []
    all_t = []
    all_g = []

    model.eval()
    with torch.no_grad():
        for t in tqdm(timesteps, desc="Computing JVP over timesteps"):
            x0 = clean_images
            t = t.expand(clean_images.shape[0])
            t = t.view(-1, 1, 1, 1).to(device)  # [B, 1, 1, 1]

            z = torch.randn_like(x0) * sigma_data
            x_t = torch.cos(t) * x0 + torch.sin(t) * z

            def model_wrapper(scaled_x_t, t):
                pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True, **model_kwargs)
                return pred, logvar

            with torch.no_grad():
                pretrain_pred = pretrained_model(x_t / sigma_data, t.flatten(), **model_kwargs)
                dxt_dt = sigma_data * pretrain_pred

            # 计算JVP
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
            v_t = torch.cos(t) * torch.sin(t)
            F_theta, F_theta_grad, _ = torch.func.jvp(model_wrapper, (x_t / sigma_data, t), (v_x, v_t), has_aux=True)

            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()

            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            second_term = -1 * torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad
            g = g + second_term

            all_F_theta_grad.append(F_theta_grad.detach())
            all_F_theta_minus.append(F_theta.detach())
            all_t.append(t.flatten())
            all_g.append(g.detach())

    # 将所有结果堆叠
    F_theta_grad = torch.stack(all_F_theta_grad)  # [num_timesteps, B, C, H, W]
    F_theta_minus = torch.stack(all_F_theta_minus)  # [num_timesteps, B, C, H, W]
    t_values = torch.stack(all_t)  # [num_timesteps, B]
    g = torch.stack(all_g)  # [num_timesteps, B, C, H, W]
    # 计算统计量
    grad_mean = F_theta_grad.mean(dim=(1, 2, 3, 4)).cpu().numpy()  # [num_timesteps]
    grad_std = F_theta_grad.std(dim=(1, 2, 3, 4)).cpu().numpy()
    minus_mean = F_theta_minus.mean(dim=(1, 2, 3, 4)).cpu().numpy()
    minus_std = F_theta_minus.std(dim=(1, 2, 3, 4)).cpu().numpy()
    t_plot = t_values[:, 0].cpu().numpy()  # 取第一个batch的时间步
    g_mean = g.mean(dim=(1, 2, 3, 4)).cpu().numpy()
    g_std = g.std(dim=(1, 2, 3, 4)).cpu().numpy()

    # 绘制统计信息
    plt.figure(figsize=(15, 5))

    # 均值和标准差
    plt.subplot(131)
    plt.plot(t_plot, grad_mean, "b-", label="F_theta_grad")
    plt.plot(t_plot, minus_mean, "r-", label="F_theta_minus")
    plt.plot(t_plot, g_mean, "g-", label="g")
    plt.fill_between(t_plot, grad_mean - grad_std, grad_mean + grad_std, color="b", alpha=0.2)
    plt.fill_between(t_plot, minus_mean - minus_std, minus_mean + minus_std, color="r", alpha=0.2)
    plt.fill_between(t_plot, g_mean - g_std, g_mean + g_std, color="g", alpha=0.2)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Mean and Std over Timesteps")

    # 标准差
    plt.subplot(132)
    plt.plot(t_plot, grad_std, "b-", label="F_theta_grad std")
    plt.plot(t_plot, minus_std, "r-", label="F_theta_minus std")
    plt.plot(t_plot, g_std, "g-", label="g std")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.title("Standard Deviation over Timesteps")

    # 差异
    plt.subplot(133)
    diff_mean = grad_mean - minus_mean
    diff_std = np.sqrt(grad_std**2 + minus_std**2)
    plt.plot(t_plot, diff_mean, "g-", label="Mean Difference")
    plt.fill_between(t_plot, diff_mean - diff_std, diff_mean + diff_std, color="g", alpha=0.2)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Difference")
    plt.legend()
    plt.title("Difference between Grad and Minus")

    plt.tight_layout()

    # 保存图像
    save_name = "scm_analysis"
    if step is not None:
        save_name += f"_step{step}"
    save_name += ".png"
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

    model.train()


def find_density_median(values, bins):
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    cdf = np.cumsum(hist) / sum(hist)  # 确保CDF正确归一化
    median_idx = np.searchsorted(cdf, 0.5)  # 寻找中位数位置
    if median_idx == len(bin_edges):
        median_value = bin_edges[-1]
    else:
        median_value = 0.5 * (bin_edges[median_idx] + bin_edges[median_idx - 1])
    return median_value


def visualize_timestep_distributions(save_dir=None, batch_size=100000):
    """
    可视化两种不同时间步长采样方案的分布

    Args:
        save_dir (str): 保存图像的目录
        batch_size (int): 采样数量
    """
    if save_dir is None:
        save_dir = "output/scm/vis_timestep_distributions"
    os.makedirs(save_dir, exist_ok=True)

    # 第一种方案：logit_normal + flow变换
    t1 = compute_density_for_timestep_sampling("logit_normal", batch_size, logit_mean=0.0, logit_std=1.0)
    flow_shift = 3.0  # 根据需要调整
    t1_shift = flow_shift * t1 / (1 + (flow_shift - 1) * t1)
    td1_t = t1_shift / (1 - t1_shift)

    t1_2 = compute_density_for_timestep_sampling("logit_normal", batch_size, logit_mean=-0.8, logit_std=1.6)
    flow_shift = 3.0  # 根据需要调整
    t1_2_shift = flow_shift * t1_2 / (1 + (flow_shift - 1) * t1_2)
    td1_2_t = t1_2_shift / (1 - t1_2_shift)

    t1_3 = compute_density_for_timestep_sampling("logit_normal", batch_size, logit_mean=0.0, logit_std=1.0)
    td1_3_t = t1_3 / (1 - t1_3)

    # 第二种方案：logit_normal_trigflow
    t2 = compute_density_for_timestep_sampling("logit_normal_trigflow", batch_size, logit_mean=-0.8, logit_std=1.6)
    tant2 = torch.tan(t2)

    t2_2 = compute_density_for_timestep_sampling("logit_normal_trigflow", batch_size, logit_mean=0.0, logit_std=1.0)
    tant2_2 = torch.tan(t2_2)

    t2_3 = compute_density_for_timestep_sampling("logit_normal_trigflow", batch_size, logit_mean=-0.4, logit_std=1.0)
    tant2_3 = torch.tan(t2_3)

    # 计算两个分布的值域范围
    min_val = min(td1_t.min().item(), tant2.min().item())
    max_val = max(td1_t.max().item(), tant2.max().item())

    density = False
    # 使用对数间隔的bins
    bins = np.logspace(np.log10(max(1e-3, min_val)), np.log10(max_val), 50)
    hist1, _ = np.histogram(td1_t.numpy(), bins=bins, density=density)
    hist2, _ = np.histogram(tant2.numpy(), bins=bins, density=density)
    hist3, _ = np.histogram(tant2_2.numpy(), bins=bins, density=density)
    hist4, _ = np.histogram(tant2_3.numpy(), bins=bins, density=density)
    hist5, _ = np.histogram(td1_2_t.numpy(), bins=bins, density=density)

    max_density = max(hist1.max(), hist2.max(), hist3.max(), hist4.max(), hist5.max())

    # 计算每个分布的均值
    mean_td1 = np.exp(np.log10(td1_t).mean().item())
    mean_td1_2 = np.exp(np.log10(td1_2_t).mean().item())
    mean_td1_3 = np.exp(np.log10(td1_3_t).mean().item())
    mean_tant2_2 = np.exp(np.log10(tant2_2).mean().item())
    mean_tant2 = np.exp(np.log10(tant2).mean().item())
    mean_tant2_3 = np.exp(np.log10(tant2_3).mean().item())

    # mean_td1 = find_density_median(td1_t.numpy(), bins)
    # mean_td1_2 = find_density_median(td1_2_t.numpy(), bins)
    # mean_tant2_2 = find_density_median(tant2_2.numpy(), bins)
    # mean_tant2 = find_density_median(tant2.numpy(), bins)
    # mean_tant2_3 = find_density_median(tant2_3.numpy(), bins)

    median_td1 = np.median(td1_t.numpy())
    median_td1_2 = np.median(td1_2_t.numpy())
    median_td1_3 = np.median(td1_3_t.numpy())
    median_tant2_2 = np.median(tant2_2.numpy())
    median_tant2 = np.median(tant2.numpy())
    median_tant2_3 = np.median(tant2_3.numpy())

    plt.figure(figsize=(12, 5))

    plt.subplot(231)
    plt.hist(td1_t.numpy(), bins=bins, density=density, alpha=0.7, label="Flow t 0/1 s3")
    plt.axvline(x=mean_td1, color="r", linestyle="--", label=f"Mean: {mean_td1:.2f}")
    plt.axvline(x=median_td1, color="g", linestyle="--", label=f"Median: {median_td1:.2f}")
    plt.xscale("log")
    plt.xlabel("t/(1-t)")
    plt.ylabel("Density")
    plt.title("Distribution of Flow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.subplot(232)
    plt.hist(td1_2_t.numpy(), bins=bins, density=density, alpha=0.7, label="Flow t -0.8/1.6 s3")
    plt.axvline(x=mean_td1_2, color="r", linestyle="--", label=f"Mean: {mean_td1_2:.2f}")
    plt.axvline(x=median_td1_2, color="g", linestyle="--", label=f"Median: {median_td1_2:.2f}")
    plt.xscale("log")
    plt.xlabel("t/(1-t)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of Flow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.subplot(233)
    plt.hist(td1_3_t.numpy(), bins=bins, density=density, alpha=0.7, label="Flow t 0/1 s1")
    plt.axvline(x=mean_td1_3, color="r", linestyle="--", label=f"Mean: {mean_td1_3:.2f}")
    plt.axvline(x=median_td1_3, color="g", linestyle="--", label=f"Median: {median_td1_3:.2f}")
    plt.xscale("log")
    plt.xlabel("t/(1-t)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of Flow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.subplot(234)
    plt.hist(tant2_2.numpy(), bins=bins, density=density, alpha=0.7, label="Trigflow t 0/1")
    plt.axvline(x=mean_tant2_2, color="r", linestyle="--", label=f"Mean: {mean_tant2_2:.2f}")
    plt.axvline(x=median_tant2_2, color="g", linestyle="--", label=f"Median: {median_tant2_2:.2f}")
    plt.xscale("log")
    plt.xlabel("tan(t)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of Trigflow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.subplot(235)
    plt.hist(tant2.numpy(), bins=bins, density=density, alpha=0.7, label="Trigflow t -0.8/1.6")
    plt.axvline(x=mean_tant2, color="r", linestyle="--", label=f"Mean: {mean_tant2:.2f}")
    plt.axvline(x=median_tant2, color="g", linestyle="--", label=f"Median: {median_tant2:.2f}")
    plt.xscale("log")
    plt.xlabel("tan(t)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of Trigflow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.subplot(236)
    plt.hist(tant2_3.numpy(), bins=bins, density=density, alpha=0.7, label="Trigflow t -0.4/1.0")
    plt.axvline(x=mean_tant2_3, color="r", linestyle="--", label=f"Mean: {mean_tant2_3:.2f}")
    plt.axvline(x=median_tant2_3, color="g", linestyle="--", label=f"Median: {median_tant2_3:.2f}")
    plt.xscale("log")
    plt.xlabel("tan(t)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of Trigflow")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "timestep_distributions_noylog_5_mean_nodensity.png"))
    plt.close()


def visualize_trigflow_distributions(save_dir=None, batch_size=100000):
    """
    可视化不同参数组合的trigflow分布。Flow单独一行，其他按std分组。

    Args:
        save_dir (str): 保存图像的目录
        batch_size (int): 采样数量
    """
    if save_dir is None:
        save_dir = "output/scm/vis_trigflow_distributions"
    os.makedirs(save_dir, exist_ok=True)

    # Flow参数固定
    t1 = compute_density_for_timestep_sampling("logit_normal", batch_size, logit_mean=0.0, logit_std=1.0)
    flow_shift = 3.0
    t1_shift = flow_shift * t1 / (1 + (flow_shift - 1) * t1)
    td1_t = t1_shift / (1 - t1_shift)

    # Trigflow参数组合
    means = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    stds = [1.0, 1.6]  # 调整顺序，先1.0后1.6

    # 存储所有分布
    all_distributions = []
    all_distributions.append(("Flow s3 0/1", td1_t))

    # 生成所有trigflow分布
    for std in stds:
        for mean in means:
            t = compute_density_for_timestep_sampling(
                "logit_normal_trigflow", batch_size, logit_mean=mean, logit_std=std
            )
            tant = torch.tan(t)
            all_distributions.append((f"Trigflow {mean}/{std}", tant))

    # 计算所有分布的值域范围
    min_val = min(dist[1].min().item() for dist in all_distributions)
    max_val = max(dist[1].max().item() for dist in all_distributions)

    # 使用对数间隔的bins
    bins = np.logspace(np.log10(max(1e-3, min_val)), np.log10(max_val), 50)

    density = False
    # 计算所有直方图的最大密度
    all_hists = []
    for _, dist in all_distributions:
        hist, _ = np.histogram(dist.numpy(), bins=bins, density=density)
        all_hists.append(hist)
    max_density = max(max(hist) for hist in all_hists)

    # 根据mean的个数确定每行的列数
    n_means = len(means)

    # 创建图像 (3行，列数根据mean的个数确定)
    plt.figure(figsize=(4 * n_means, 12))

    # 先画Flow (第一行中间)
    plt.subplot(3, n_means, n_means // 2 + 1)  # 放在第一行中间位置
    name, dist = all_distributions[0]  # Flow的分布
    plt.hist(dist.numpy(), bins=bins, density=density, alpha=0.7, label=name)
    mean = np.exp(np.log10(dist).mean().item())
    median = np.median(dist.numpy())
    plt.axvline(x=mean, color="r", linestyle="--", label=f"Mean: {mean:.2f}")
    plt.axvline(x=median, color="g", linestyle="--", label=f"Median: {median:.2f}")
    plt.xscale("log")
    plt.xlabel("t/(1-t)")
    plt.ylabel("Count")
    plt.title(name)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlim(min_val, max_val)
    plt.ylim(0, max_density * 1.1)

    # 画Trigflow分布 (第二行std=1.0，第三行std=1.6)
    for std_idx, std in enumerate(stds):
        for mean_idx, mean in enumerate(means):
            dist_idx = 1 + std_idx * len(means) + mean_idx  # 跳过Flow
            name, dist = all_distributions[dist_idx]

            plt.subplot(3, n_means, n_means * (std_idx + 1) + mean_idx + 1)
            plt.hist(dist.numpy(), bins=bins, density=density, alpha=0.7, label=name)
            mean_val = np.exp(np.log10(dist).mean().item())
            median = np.median(dist.numpy())
            plt.axvline(x=mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.2f}")
            plt.axvline(x=median, color="g", linestyle="--", label=f"Median: {median:.2f}")
            plt.xscale("log")
            plt.xlabel("tan(t)")
            plt.ylabel("Count")
            plt.title(name)
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.xlim(min_val, max_val)
            plt.ylim(0, max_density * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trigflow_distributions.png"))
    plt.close()


if __name__ == "__main__":
    # visualize_timestep_distributions()
    visualize_trigflow_distributions()
