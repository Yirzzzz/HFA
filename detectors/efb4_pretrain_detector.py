import os
import datetime
import logging
import time
from torch.distributions import Categorical, Normal
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import math
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors import EfficientDetector
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

@torch.no_grad()
def tsne_plot_multi(features: torch.Tensor, labels: torch.Tensor,
                           save_path: str = None, class_names: dict = None,
                           max_points_per_class: int = 2000):
    """
    高级版 t-SNE 可视化，带顶会风格美化。
    features: [B,C,H,W]
    labels:   [B]
    class_names: {int: str}，例如 {0:'real',1:'fake',2:'fake_aug'}
    """
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("whitegrid")

    feat = features.detach().float().cpu()
    y = labels.detach().long().cpu().numpy()
    X = feat.mean(dim=(2,3)).numpy().astype('float32')  # [B,C]

    # 每类子采样
    idx_all = []
    for cls in np.unique(y):
        idx_c = np.where(y == cls)[0]
        if len(idx_c) == 0: continue
        if len(idx_c) > max_points_per_class:
            idx_c = np.random.choice(idx_c, max_points_per_class, replace=False)
        idx_all.append(idx_c)
    if len(idx_all) == 0:
        print("[t-SNE] 所有类别都为空，跳过绘制。")
        return
    idx_all = np.concatenate(idx_all, 0)
    X, y = X[idx_all], y[idx_all]

    n_samples, n_features = X.shape
    if n_samples < 3:
        print(f"[t-SNE] 样本太少 (n={n_samples})，跳过绘制。")
        return

    # PCA & t-SNE
    pca_dims = int(min(50, n_features, max(2, n_samples - 1)))
    Xp = PCA(n_components=pca_dims, random_state=0).fit_transform(X)
    perp = int(max(2, min(30, n_samples - 1, n_samples // 3)))
    Z = TSNE(n_components=2, init='pca', learning_rate='auto',
             perplexity=perp, n_iter=1500, random_state=0).fit_transform(Xp)

    # 绘制
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    palette = sns.color_palette("Set2", n_colors=len(np.unique(y)))
    markers = ['o', 's', 'D', '^', 'P', 'X', '*', 'v', '<', '>']

    for i, cls in enumerate(np.unique(y)):
        m = (y == cls)
        name = class_names.get(int(cls), f"class-{int(cls)}") if class_names else f"class-{int(cls)}"
        ax.scatter(Z[m, 0], Z[m, 1],
                   s=22, alpha=0.85,
                   color=palette[i % len(palette)],
                   edgecolor='white', linewidth=0.4,
                   marker=markers[i % len(markers)],
                   label=name)

    # 图例与样式
    leg = ax.legend(markerscale=1.5, fontsize=10, frameon=True, loc='best')
    leg.get_frame().set_edgecolor('lightgray')
    leg.get_frame().set_alpha(0.9)
    ax.set_title(f"t-SNE visualization (n={n_samples}, perp={perp}, pca_dims={pca_dims})",
                 fontsize=12, pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


class TSNEBuffer:
    def __init__(self, max_per_class=4000, seed=0):
        self.max_per_class = max_per_class
        self.store = defaultdict(list)
        np.random.seed(seed)

    @torch.no_grad()
    def add(self, features: torch.Tensor, labels: torch.Tensor):
        # GAP -> [B,C]
        feat = features.detach().float().mean(dim=(2,3)).cpu().numpy()
        y = labels.detach().long().cpu().numpy()
        for xi, yi in zip(feat, y):
            L = self.store[int(yi)]
            L.append(xi)
            # 截断到每类 max_per_class（简单随机下采样）
            if len(L) > self.max_per_class:
                idx = np.random.choice(len(L), self.max_per_class, replace=False)
                self.store[int(yi)] = [L[i] for i in idx]

    def flush_arrays(self):
        if not any(len(v) > 0 for v in self.store.values()):
            return None, None
        X_list, y_list = [], []
        for k, arr in self.store.items():
            if len(arr) == 0: continue
            X_list.append(np.stack(arr, 0))
            y_list.append(np.full((len(arr),), k, dtype=np.int64))
        X = np.concatenate(X_list, 0).astype('float32')
        y = np.concatenate(y_list, 0)
        return X, y

class LatentAug(nn.Module):
    """
    两种增强（集成版）：
      - Cayley 轻旋：针对低能样本（结构小扰动）
      - Channel Dropout：针对高能样本（抑制过拟合）
    模式：
      - mode='random'  : 随机在两者间选择（对照）
      - mode='energy'  : 用 fake_factor 的能量分流 + 强度门控
    输入：
      - random:  aug(fake_feat)
      - energy:  aug(fake_feat, fake_factor)  # 能量由 fake_factor 计算
    输出：
      - aug_feat（可选返回 op_idx: 0=Cayley, 1=Dropout）
    """
    def __init__(self,
                 mode: str = 'energy',           # 'random' | 'energy'
                 use_log_energy: bool = True,    # 对能量取 log，稳住极小值
                 ema_momentum: float = 0.99,          # 'mean' | 'median' | float(绝对阈值，基于 e 或 log e)
                 cayley_eps: float = 0.001,
                 cayley_subq: int = 64,
                 drop_prob: float = 0.30,
                 apply_eval: bool = False,
                 config = None):      # eval() 时是否也增强
        super().__init__()
        assert mode in ('random', 'energy')
        self.mode = mode
        self.use_log_energy = bool(use_log_energy)
        self.ema_momentum = float(ema_momentum)
        self.cayley_eps = float(cayley_eps)
        self.cayley_subq = int(cayley_subq)
        self.drop_prob = float(drop_prob)
        self.apply_eval = bool(apply_eval)
        self.fda = FeatureDiscriminativeAugLite()
        self.config = config
        # 随机模式下两种操作的概率（[Cayley, Dropout]）
        self.register_buffer('p_ops2', torch.tensor([0.5, 0.5], dtype=torch.float))
        # 初始值设为0和1，它们会在第一个训练步长后迅速调整到合理范围
        self.register_buffer('q1_ema', torch.tensor(0.0))
        self.register_buffer('q3_ema', torch.tensor(1.0))
        self.register_buffer('q_ema_inited', torch.tensor(False, dtype=torch.bool))
        self.eps_q = 1e-6  # 分位间隔最小保护
        self.dropBlock = DropBlock(drop_prob=drop_prob)
        self.dropout = VanillaFeatureDropout(p=drop_prob)
        self.channel_dropout = ChannelDropout(drop_prob=drop_prob)

    # ---------- utils ----------
    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.flatten(1).pow(2).mean(1).sqrt()  # [B]

    def _energy(self, feat_for_energy: torch.Tensor) -> torch.Tensor:
        e = self._rms(feat_for_energy.float()).clamp_min(1e-12)        # [B]
        return torch.log(e) if self.use_log_energy else e               # log 标度可拉开“地板”分辨率

    def spatial_affine_warp(self, feat, angle_std_deg=2.0, trans_std=0.05):
        B, _, H, W = feat.shape
        # 角度扰动
        angle = torch.randn(B, device=feat.device) * (angle_std_deg / 180.0 * 3.14159)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        # 平移扰动
        trans_x = torch.randn(B, device=feat.device) * trans_std
        trans_y = torch.randn(B, device=feat.device) * trans_std

        # 构造仿射矩阵
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, trans_x], dim=1),
            torch.stack([sin_a, cos_a, trans_y], dim=1)
        ], dim=1)  # Shape: [B, 2, 3]

        grid = F.affine_grid(theta, feat.size(), align_corners=False)
        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    @torch.no_grad()
    def centrifugal_direct(self, z, beta=None):
        B = z.size(0)
        mu = z.mean(dim=0, keepdim=True)
        if beta is None:
            beta = torch.rand(B, 1, 1, 1, device=z.device)
        return z + beta * (z - mu)

    @torch.no_grad()
    def centrifugal_to_hardest(self, z, beta=None):
        B = z.size(0)
        mu = z.mean(dim=0, keepdim=True)
        d = ((z - mu) ** 2).flatten(1).sum(1)  # [B]
        idx = torch.argmax(d)  # 最难样本 a
        a = z[idx:idx + 1]  # [1,C,H,W]
        if beta is None:
            beta = torch.rand(B, 1, 1, 1, device=z.device)
        return z + beta * (a - z)

    # ==== 3) Affine（小角度旋转）====
    def affine_rotate(self,z , theta_deg=10.0):
        B, C, H, W = z.shape
        theta = (torch.rand(B, device=z.device) * 2 - 1) * (theta_deg / 180.0 * 3.1415926)  # [-θ, θ]
        cos, sin = torch.cos(theta), torch.sin(theta)
        A = torch.zeros(B, 2, 3, device=z.device)
        A[:, 0, 0] = cos;
        A[:, 0, 1] = -sin
        A[:, 1, 0] = sin;
        A[:, 1, 1] = cos
        grid = F.affine_grid(A, size=z.size(), align_corners=False)
        z_rot = F.grid_sample(z, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return z_rot

    # ==== 4) Additive（潜空间 GMM 噪声）====
    def additive_gmm(self, z, pis=(0.6, 0.4), sigmas=(0.05, 0.15)):
        B = z.size(0)
        cat = Categorical(probs=torch.tensor(pis, device=z.device))
        k = cat.sample((B,))  # [B] 选择混合c分量
        sigma = torch.tensor(sigmas, device=z.device)[k].view(B, 1, 1, 1)
        eps = torch.randn_like(z) * sigma
        return z + eps

    def calc_mean_std(self, feat, eps=1e-5):
        """
        计算特征的 Channel-wise 均值和标准差
        feat: [N, C, H, W]  <-- 支持 Batch 操作
        """
        N, C, H, W = feat.size()
        # var: [N, C] -> [N, C, 1, 1]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        # mean: [N, C] -> [N, C, 1, 1]
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain_mix_augmentation(self, z_c, z_s, alpha_min=0.5, alpha_max=1.0):
        """
        z_c: content features [N_c, C, H, W]  (e.g., ambiguous/core)
        z_s: style  features [N_s, C, H, W]  (e.g., core/sat/cross-domain)
        返回: z_fused [N_c, C, H, W]，保证“content 这边一个不落下”
        """
        # 1) 空集保护
        if z_c is None or z_s is None:
            return z_c
        if z_c.numel() == 0 or z_s.numel() == 0:
            return z_c

        # 2) 只要求 (C, H, W) 相同
        assert z_c.size()[1:] == z_s.size()[1:], \
            f"Shape mismatch: z_c={z_c.size()}, z_s={z_s.size()}"

        N_c, C, H, W = z_c.size()
        N_s = z_s.size(0)

        # 3) 为每个 content 样本分配一个 style 索引
        if N_s >= N_c:
            # 无放回采样
            perm = torch.randperm(N_s, device=z_s.device)[:N_c]
        else:
            # 有放回采样：让 style “循环使用”
            perm = torch.randint(low=0, high=N_s, size=(N_c,), device=z_s.device)

        z_s_matched = z_s[perm]  # [N_c, C, H, W]

        # 4) 计算统计量
        mu_c, std_c = self.calc_mean_std(z_c)  # [N_c, C, 1, 1]
        mu_s, std_s = self.calc_mean_std(z_s_matched)

        # 5) alpha: 每个样本一个 alpha
        alpha = torch.empty((N_c, 1, 1, 1), device=z_c.device).uniform_(alpha_min, alpha_max)

        mu_mix = alpha * mu_c + (1 - alpha) * mu_s
        std_mix = alpha * std_c + (1 - alpha) * std_s

        # 6) AdaIN
        z_c_norm = (z_c - mu_c) / (std_c + 1e-6)
        z_fused = z_c_norm * std_mix + mu_mix

        return z_fused

    @torch.no_grad()
    def blend_features_batch(self, f1, f2, method='slerp'):
        """
        f1: [B1, C, H, W]
        f2: [B2, C, H, W]
        返回:
            fused: [B1*B2, C, H, W]
        """
        B1, C, H, W = f1.shape
        B2 = f2.shape[0]
        fused_list = []
        if method == 'onebyoneSlerp':
            for i in range(B1):
                f1_sample = f1[i]
                f2_sample = f2[i]
                f1_flat = F.normalize(f1_sample.view(-1), dim=0)
                f2_flat = F.normalize(f2_sample.view(-1), dim=0)
                dot = torch.clamp(torch.dot(f1_flat, f2_flat), -1.0, 1.0)
                omega = torch.acos(dot)
                if torch.abs(omega) < 1e-6:
                    fused_flat = f1_flat
                else:
                    alpha = torch.empty((), device=f1.device).uniform_(0.85, 0.99)
                    fused_flat = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * f1_flat + \
                                 (torch.sin(alpha * omega) / torch.sin(omega)) * f2_flat
                fused = fused_flat.view(C, H, W)
                fused_list.append(fused.unsqueeze(0))
        else:
            for i in range(B1):
                for j in range(B2):
                    f1_sample = f1[i]
                    f2_sample = f2[j]

                    if method == "linear":
                        # 连续采样 α ∈ [0.7, 0.9]
                        alpha = torch.empty((), device=f1.device).uniform_(0, 1)
                        fused = alpha * f1_sample + (1 - alpha) * f2_sample

                    elif method == "slerp":
                        f1_flat = F.normalize(f1_sample.view(-1), dim=0)
                        f2_flat = F.normalize(f2_sample.view(-1), dim=0)
                        dot = torch.clamp(torch.dot(f1_flat, f2_flat), -1.0, 1.0)
                        omega = torch.acos(dot)
                        if torch.abs(omega) < 1e-6:
                            fused_flat = f1_flat
                        else:
                            alpha = torch.empty((), device=f1.device).uniform_(0, 1)
                            fused_flat = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * f1_flat + \
                                         (torch.sin(alpha * omega) / torch.sin(omega)) * f2_flat
                        fused = fused_flat.view(C, H, W)

                    else:
                        raise ValueError(f"Unknown blend method: {method}")

                    fused_list.append(fused.unsqueeze(0))

        fused_batch = torch.cat(fused_list, dim=0) if len(fused_list) > 0 else f1.new_zeros((0, C, H, W))
        return fused_batch
    # def _cayley_jitter(self, feat: torch.Tensor, eps: float = None, sub_q: int = None) -> torch.Tensor:
    #     """Cayley 轻旋扰动（不求梯度，作为数据增强）"""
    #     if eps is None:   eps = self.cayley_eps
    #     if sub_q is None: sub_q = self.cayley_subq
    #     B, C, H, W = feat.shape
    #     if B < 2:
    #         return feat
    #     X  = feat.view(B, -1)                    # [B, D]
    #     D  = X.size(1)
    #     mu = X.mean(0, keepdim=True)
    #     Xc = X - mu
    #     q  = int(min(sub_q, B - 1, D))
    #     if q < 2:
    #         return feat
    #     A = torch.randn(q, q, device=X.device); A = A - A.T
    #     I = torch.eye(q, device=X.device)
    #     R = torch.linalg.solve(I - 0.5*eps*A, I + 0.5*eps*A)  # q×q
    #     P = torch.randn(D, q, device=X.device)
    #     Xq = Xc @ P
    #     Xq_rot = Xq @ R.T
    #     Xc_hat = Xq_rot @ P.T
    #     X_hat = mu + Xc_hat
    #     return X_hat.view(B, C, H, W)
    def _cayley_jitter(self,
                       feat: torch.Tensor,
                       eps: float = None,
                       sub_q: int = None) -> torch.Tensor:
        """
        Cayley 轻旋（Type-A用）
        - 在随机 q 维子空间做正交旋转，正交补保持恒等
        - 近似能量保持 & 结构保持；适合“中间区域”
        """
        if eps is None:   eps = getattr(self, "cayley_eps", 0.01)
        if sub_q is None: sub_q = getattr(self, "cayley_subq", 64)

        B, C, H, W = feat.shape
        if B == 0 or eps <= 0: return feat

        X = feat.view(B, -1)  # [B, D], D=C*H*W
        D = X.size(1)
        # q = int(np.random.uniform(128, 256))
        q = 256
        if q < 2: return feat

        # 1) 中心化
        mu = X.mean(0, keepdim=True)  # [1, D]
        Xc = X - mu  # [B, D]

        # 2) 构造正交子空间基 P (D×q), P^T P = I_q
        P = torch.randn(D, q, device=X.device)
        P, _ = torch.linalg.qr(P, mode='reduced')  # 正交化

        # 3) 子空间 Cayley 旋转 R (q×q)
        A = torch.randn(q, q, device=X.device)
        A = A - A.T
        Iq = torch.eye(q, device=X.device)
        R = torch.linalg.solve(Iq - 0.5 * eps * A, Iq + 0.5 * eps * A)  # 正交

        # 4) 低秩更新：X_rot = Xc + (Xc P)(R - I)P^T + mu
        XP = Xc @ P  # [B, q]
        delta = XP @ (R - Iq) @ P.T  # [B, D]
        Xrot = Xc + delta + mu  # [B, D]

        return Xrot.view(B, C, H, W)

    @torch.no_grad()
    def _channel_dropout(self, x: torch.Tensor, drop_prob: float = None, rescale: bool = True) -> torch.Tensor:
        """通道级 dropout（空间共享掩码 + 轻噪声）"""
        if drop_prob is None: drop_prob = self.drop_prob
        B, C, H, W = x.shape
        if B == 0 or drop_prob <= 0.0:
            return x
        mask = (torch.rand(B, C, 1, 1, device=x.device) > drop_prob).to(x.dtype)  # [B,C,1,1]
        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise  # <<< 修复：真正加入噪声
        x_aug = x_noisy * mask
        if rescale:
            keep_ratio = mask.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            x_aug = x_aug / keep_ratio
        return x_aug

    @torch.no_grad()
    def _update_q_ema(self, q1_batch: torch.Tensor, q3_batch: torch.Tensor):
        if not bool(self.q_ema_inited.item()):
            # warm-start：首批直接拷贝，避免 q3≈1 的偏置
            self.q1_ema.copy_(q1_batch)
            self.q3_ema.copy_(q3_batch)
            self.q_ema_inited.fill_(True)
            return
        m = self.ema_momentum
        # 原地更新，少一次张量重绑定
        self.q1_ema.mul_(m).add_((1 - m) * q1_batch)
        self.q3_ema.mul_(m).add_((1 - m) * q3_batch)

    def _gap(self, x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B,C,H,W) → (B,C)

    @torch.no_grad()
    def _pick_nearest_real(self,
                           real_feat: torch.Tensor,  # [Br,C,H,W]
                           fake_feat: torch.Tensor,  # [Bf,C,H,W]
                           metric: str = 'maha',  # 'maha' | 'cos'
                           mu_r: torch.Tensor = None,  # [C]  真实原型均值(EMA)
                           var_r: torch.Tensor = None  # [C]  真实方差(EMA, 对角)
                           ):
        """
        返回:
          nearest_real: [Bf,C,H,W]  每个 fake 对应的最近 real 特征
          idx         : [Bf]        对应在 real_feat 中的索引
          score       : [Bf]        相似度或距离(方便日志)
            - metric='cos'  -> 最大余弦相似度
            - metric='maha' -> 最小马氏距离
        """
        if real_feat is None or real_feat.numel() == 0 or fake_feat is None or fake_feat.numel() == 0:
            return None, None, None

        # GAP 到向量空间
        h_r = self._gap(real_feat).contiguous()  # [Br,C]
        h_f = self._gap(fake_feat).contiguous()  # [Bf,C]

        if metric == 'maha' and (mu_r is not None) and (var_r is not None):
            inv_std = 1.0 / (var_r.sqrt() + 1e-6)  # [C]
            h_r_w = (h_r - mu_r) * inv_std  # [Br,C]
            h_f_w = (h_f - mu_r) * inv_std  # [Bf,C]
            # 欧氏距离在白化空间 = 对角马氏距离
            diff = h_f_w[:, None, :] - h_r_w[None, :, :]  # [Bf,Br,C]
            dist = (diff * diff).sum(dim=-1).sqrt()  # [Bf,Br]
            idx = dist.argmin(dim=1)  # [Bf]
        else:
            # 余弦最近邻（与你的实现等价）
            h_r_n = F.normalize(h_r, dim=1)
            h_f_n = F.normalize(h_f, dim=1)
            sims = torch.matmul(h_f_n, h_r_n.t())  # [Bf,Br]
            idx = sims.argmax(dim=1)  # [Bf]

        nearest_real = real_feat.index_select(0, idx)  # [Bf,C,H,W]
        return nearest_real

    # ---------- forward ----------
    def forward(self, fake_feat: torch.Tensor, real_feat: torch.Tensor, fake_factor: torch.Tensor = None,
                return_op_idx: bool = False):
        """
        fake_feat   : [B,C,H,W]  —— 需增强的特征
        fake_factor : [B,C,H,W]  —— 仅在 mode='energy' 时需要；用它来计算能量
        return_op_idx: 返回每个样本使用的增强编号（0=Cayley, 1=Dropout）
        """
        assert fake_feat.dim() == 4, "fake_feat 应为 [B,C,H,W]"
        if (not self.training) and (not self.apply_eval):
            return (fake_feat, None) if return_op_idx else fake_feat

        B = fake_feat.size(0)
        if B == 0:
            return (fake_feat, None) if return_op_idx else fake_feat

        z = fake_feat

        # ------- A) random（对照） -------
        if self.mode == 'random':
            with torch.no_grad():
                op_idx = torch.multinomial(self.p_ops2.to(z.device), num_samples=B, replacement=True)  # 0/1
            out = z.clone()
            # 0: Cayley
            sel = (op_idx == 0)
            if sel.any():
                out[sel] = self._cayley_jitter(z[sel])
            # 1: Dropout
            sel = (op_idx == 1)
            if sel.any():
                out[sel] = self._channel_dropout(z[sel], drop_prob=self.drop_prob, rescale=True)
            return (out, op_idx) if return_op_idx else out

        # ------- B) energy（用 fake_factor 分流 + 强度门控） -------
        assert fake_factor is not None, "mode='energy' 需要提供 fake_factor"
        with torch.no_grad():
            e_use = self._energy(fake_factor)  # [B]

            if self.training and e_use.numel() > 1:
                q1_b = torch.quantile(e_use,  0.15)
                q3_b = torch.quantile(e_use, 0.7)
                # 只有当 batch 分位有效时才更新 EMA（防止同质批把分位夹死）
                if (q3_b - q1_b) >= self.eps_q:
                    self._update_q_ema(q1_b, q3_b)
            q1 = self.q1_ema
            q3 = self.q3_ema
            # 若仍退化，给个最小间隔，且尽量维持上一轮的相对关系
            if (q3 - q1) < self.eps_q:
                q3 = q1 + self.eps_q

            # 严格三段：<Q1、[Q1,Q3)、>=Q3
            low_mask = (e_use < q1)
            high_mask = (e_use >= q3)
            mid_mask = (~low_mask) & (~high_mask)

        out = z.clone()
        aug_mask = torch.zeros(B, dtype=torch.bool, device=z.device)


        # # 中段：保持你原逻辑
        # if mid_mask.any():
        #     out[mid_mask] = self._cayley_jitter(z[mid_mask])
        low_feat = None
        mid_feat = None
        high_feat = None

        # 高能：记得回写
        if high_mask.any():
            high_feat = out[high_mask]

            # if  self.config['blend_mode'] == 'dropout':
            #     aug_high_feat = self.dropout(high_feat)
            # elif self.config['blend_mode'] == 'channel':
            #     aug_high_feat = self.channel_dropout(high_feat)
            # else:
            #     aug_high_feat = self.dropBlock(high_feat)
            aug_high_feat = self._cayley_jitter(high_feat)
            out[high_mask] = aug_high_feat
            aug_mask[high_mask] = True

        if low_mask.any():
            low_feat = out[low_mask]
            # aug_mid_feat = self.additive_gmm(low_feat)
            # out[low_mask] = aug_mid_feat
            # aug_mask[low_mask] = True
        if mid_mask.any():
            mid_feat = out[mid_mask]
            aug_mid_feat = self._cayley_jitter(mid_feat)
            out[mid_mask] = aug_mid_feat
            aug_mask[mid_mask] = True

        res = out[aug_mask]
        if self.config['blend_mode'] == 'mid':
            if low_feat is not None and mid_feat is not None :  # 注意用 size(0)
                aug_low_feat = self.blend_features_batch(mid_feat, mid_feat, method=self.config['aug_mode'])
                res = torch.concat((res, aug_low_feat), dim=0)



        if return_op_idx:
            op_idx = torch.full((B,), -1, dtype=torch.long, device=z.device)
            op_idx[mid_mask] = 0
            op_idx[high_mask] = 1
            return res , op_idx
        return res


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelDropout(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if (not self.training) or self.drop_prob <= 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        B, C, _, _ = x.shape

        # 用 float32 采样更稳，然后 cast 到 x.dtype
        rand = torch.rand((B, C, 1, 1), device=x.device, dtype=torch.float32)
        mask = (rand < keep_prob).to(dtype=x.dtype)  # 0/1

        # 经典 dropout 缩放：保持期望不变
        return x * mask / keep_prob

class VanillaFeatureDropout(nn.Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(p=p)

    def forward(self, x):
        # x: [B,C,H,W] -> element-wise dropout
        return self.drop(x)

class DropBlock(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks.

    论文地址 (Paper URL): https://arxiv.org/abs/1810.12890
    """

    def __init__(self, block_size: int = 3, drop_prob: float = 0.015):
        """
        Args:
            block_size (int): 要丢弃的块的大小。必须是奇数。
            drop_prob (float): 丢弃一个单元格的概率。这是总体的丢弃率，而不是每个位置的丢弃率。
        """
        super(DropBlock, self).__init__()

        if not isinstance(block_size, int) or block_size % 2 != 1:
            raise ValueError("block_size 必须是奇数整数。")

        self.block_size = 3
        self.drop_prob = drop_prob
        # 计算用于 max_pool2d 的 padding
        self.padding = block_size // 2

    def _calculate_gamma(self, feat_size: tuple) -> float:
        """
        根据原文公式计算 gamma，即中心点被选为 block 中心的概率。
        """
        h, w = feat_size

        # 可放置 block 的有效区域大小
        # 例如 8x8 特征图，block_size=3，有效区域是 (8-3+1)x(8-3+1) = 6x6
        valid_area = (h - self.block_size + 1) * (w - self.block_size + 1)

        # 期望要丢弃的单元格总数
        num_expected_to_drop = self.drop_prob * (h * w)

        # 每个 block 包含的单元格数
        block_area = self.block_size ** 2

        # gamma = (期望丢弃数 / 每个 block 面积) / 可放置 block 的总位置数
        gamma = (num_expected_to_drop / block_area) / valid_area
        return gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        b, c, h, w = x.shape

        # 1. 计算 gamma
        gamma = self._calculate_gamma((h, w))

        # 2. 生成一个伯努利掩码 (seed mask)
        # 掩码形状为 [B, 1, H, W]，在所有通道上共享，更高效
        mask_shape = (b, 1, h, w)
        # torch.rand 返回 [0, 1) 的均匀分布随机数
        # 小于 gamma 的位置为 True (1)，表示这里是 block 的中心点
        seed_mask = (torch.rand(mask_shape, device=x.device) < gamma)

        # 3. 利用 max_pooling 将中心点的 1 扩展为 block_size 大小的块
        # 这是原文中提到的高效实现技巧，相当于形态学中的“膨胀”操作
        block_mask = F.max_pool2d(
            seed_mask.float(),
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.padding
        )

        # 4. 将 block_mask 反转，0 表示丢弃，1 表示保留
        keep_mask = 1.0 - block_mask

        # 5. 应用掩码
        out = x * keep_mask

        # 6. 标准化（非常重要）
        # 为了在训练和推理时保持特征的期望值一致，需要将保留的元素进行放大
        # 放大比例 = 总元素数 / 保留的元素数
        num_kept = torch.sum(keep_mask)
        num_total = keep_mask.numel()
        scale = num_total / num_kept.clamp_min(1.0)  # clamp_min(1.0) 防止除以零

        return out * scale
class ForgeryUNet(nn.Module):
    """
    Lightweight Forgery Factor Extractor (FFE):
    - 3 Conv-BN-ReLU encoder blocks (stride 1,2,2)
    - 2 deconv upsampling layers
    - 1x1 Conv + Tanh head
    - spatial mean-normalization to enforce zero-mean residual map
    """
    def __init__(self, in_ch=1792, base=128, mode='mask_res'):
        super().__init__()
        self.mode = mode

        # encoder: Conv + BN + ReLU
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.GroupNorm(8, base),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base, 3, 2, 1),
            nn.GroupNorm(8, base),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base, 2 * base, 3, 2, 1),
            nn.GroupNorm(16, 2*base),
            nn.ReLU(inplace=True),
        )

        # decoder: lightweight upsampling
        self.dec2 = nn.ConvTranspose2d(2 * base, base, 2, 2)
        self.dec1 = nn.ConvTranspose2d(base, base, 2, 2)

        # head: 1x1 conv + Tanh
        self.head_r = nn.Sequential(
            nn.Conv2d(base, in_ch, 1),
            nn.Tanh(),
        )

    def forward(self, x, return_mask=False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d1 = self.dec1(d2)

        R = self.head_r(d1)
        # zero-mean across spatial dims to suppress content bias
        R = R - R.mean(dim=(2, 3), keepdim=True)

        if return_mask:
            return R, None
        return R

# class ForgeryUNet(nn.Module):
#     def __init__(self, in_ch=1792, base=128, mode='mask_res'):
#         super().__init__()
#         self.mode = 'mask_res'
#         self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, 1, 1), nn.ReLU(inplace=True))
#         self.enc2 = nn.Sequential(nn.Conv2d(base, base, 3, 2, 1), nn.ReLU(inplace=True))
#         self.enc3 = nn.Sequential(nn.Conv2d(base, 2*base, 3, 2, 1), nn.ReLU(inplace=True))
#         self.dec2 = nn.ConvTranspose2d(2*base, base, 2, 2)
#         self.dec1 = nn.ConvTranspose2d(base, base, 2, 2)
#         self.head_m = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())
#         self.head_r = nn.Sequential(nn.Conv2d(base, in_ch, 1), nn.Tanh())
#
#     def forward(self, x, return_mask=False):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         d2 = self.dec2(e3)
#         d1 = self.dec1(d2)
#         # M = self.head_m(d1)      # 先不用
#         R = self.head_r(d1)
#         R = R - R.mean(dim=(2, 3), keepdim=True)
#         O = R  # 直接用 R
#         if return_mask:
#             return O, None
#         return O

class FakeFeatRandomDropout(nn.Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if (not self.training) or self.drop_prob <= 0.0:
            return x

        B, C, H, W = x.shape

        channel_mask = (torch.rand(B, C, 1, 1, device=x.device) > self.drop_prob).float()
        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise
        x_aug = x_noisy * channel_mask
        keep_ratio = channel_mask.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        x_aug = x_aug / keep_ratio
        return x_aug
#
# class FakeFeatRandomDropout(nn.Module):
#     def __init__(self, drop_prob=0.3):
#         """
#         直接对 fake_feat 进行随机 dropout，保持输入输出形状
#         drop_prob: 丢弃概率 ∈ [0, 1)，建议 0.1 ~ 0.5 之间
#         """
#         super().__init__()
#         self.drop_prob = drop_prob
#         self.dropout = nn.Dropout2d(p=drop_prob, inplace=False)
#
#     def forward(self, x):
#         """
#         输入：x ∈ [B, 1792, 8, 8]
#         输出：同样形状的特征（部分被置零）
#         """
#         if  self.drop_prob <= 0.0:
#             return x
#         return self.dropout(x)
class EnergyGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha1_param = nn.Parameter(torch.tensor(0.0))  # sigmoid控制在合理区间
        self.alpha2_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, mu_real, std_real, mu_fake, std_fake):
        alpha1 = torch.sigmoid(self.alpha1_param) * 3  # ∈ (0, 3)
        alpha2 = torch.sigmoid(self.alpha2_param) * 3
        min_thresh = mu_real + std_real * alpha1
        max_thresh = mu_fake - std_fake * alpha2
        return min_thresh, max_thresh

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDiscriminativeAugLite(nn.Module):
    """
    输入 feat:[B,C,H,W] -> 输出增强 feat_aug:[B,C,H,W]
    - 无需分类头或外部能量函数
    - 通道重要性: 能量(|x|的均值) + 空间梯度强度 的加权
    - 高重要通道: 弱扰动/保护；低重要通道: 强扰动 + 随机drop
    - 可选空间细化: 把增强集中在弱响应区域
    """
    def __init__(self,
                 topk_ratio: float = 0.25,   # 保护前25%重要通道
                 strong_sigma: float = 0.20, # 低重要通道强扰动
                 weak_sigma: float = 0.05,   # 高重要通道弱扰动
                 p_drop_low: float = 0.15,   # 低重要通道随机drop概率
                 use_spatial_refine: bool = True,  # 是否做空间细化
                 energy_weight: float = 0.5, # 通道能量权重
                 grad_weight: float = 0.5,   # 空间梯度权重
                 ):
        super().__init__()
        assert 0.0 <= topk_ratio < 1.0
        self.topk_ratio = topk_ratio
        self.strong_sigma = strong_sigma
        self.weak_sigma = weak_sigma
        self.p_drop_low = p_drop_low
        self.use_spatial_refine = use_spatial_refine
        self.energy_weight = energy_weight
        self.grad_weight = grad_weight
        if use_spatial_refine:
            self.spatial_refine = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        else:
            self.spatial_refine = None

    @staticmethod
    def _minmax_norm(x, eps=1e-6):
        xmin = x.amin(dim=1, keepdim=True)
        xmax = x.amax(dim=1, keepdim=True)
        return (x - xmin) / (xmax - xmin + eps)

    @torch.no_grad()
    def _channel_importance(self, feat: torch.Tensor):
        """
        无监督通道重要性（每个样本独立计算）:
        score = w_e * norm(mean|x|) + w_g * norm(mean(|dx|+|dy|))
        返回: [B,C] in [0,1]
        """
        # 通道能量
        energy = feat.abs().mean(dim=(2,3))  # [B,C]
        # 空间梯度
        dx = feat[:,:,:,1:] - feat[:,:,:,:-1]
        dy = feat[:,:,1:,:] - feat[:,:,:-1,:]
        dx = F.pad(dx, (0,1,0,0))
        dy = F.pad(dy, (0,0,0,1))
        grad_mag = (dx.abs() + dy.abs()).mean(dim=(2,3))  # [B,C]

        e_n = self._minmax_norm(energy)
        g_n = self._minmax_norm(grad_mag)
        score = self.energy_weight * e_n + self.grad_weight * g_n
        score = score / (score.amax(dim=1, keepdim=True).clamp_min(1e-6))
        return score

    def forward(self, feat: torch.Tensor):
        """
        训练态增强；eval() 直接透传
        """
        if not self.training:
            return feat

        B, C, H, W = feat.shape
        device = feat.device

        # 1) 通道重要性 & top-k 保护（用 topk + scatter 更稳）
        imp_ch = self._channel_importance(feat)           # [B,C]
        k = max(1, int(self.topk_ratio * C))
        topk_idx = torch.topk(imp_ch, k, dim=1, largest=True, sorted=False).indices  # [B,k]
        protect_mask_ch = torch.zeros(B, C, device=device)
        protect_mask_ch.scatter_(1, topk_idx, 1.0)        # 高重要=1
        enhance_mask_ch = 1.0 - protect_mask_ch

        # 2) 广播到空间
        protect_mask = protect_mask_ch.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        enhance_mask = 1.0 - protect_mask                           # [B,C,1,1]

        # 3) 空间细化（可选）：弱响应位置放大增强
        if self.use_spatial_refine:
            resp = feat.pow(2).mean(1, keepdim=True)  # [B,1,H,W]
            resp = resp / (resp.amax(dim=(2,3), keepdim=True).clamp_min(1e-6))
            spatial_weight = 1.0 - resp
            spatial_weight = self.spatial_refine(spatial_weight).sigmoid()  # [B,1,H,W]
            enhance_mask = enhance_mask * spatial_weight
            protect_mask = 1.0 - (1.0 - protect_mask) * spatial_weight

        # 4) 低重要通道强扰动 + 随机drop；高重要通道弱扰动
        noise_strong = torch.randn_like(feat) * self.strong_sigma
        noise_weak   = torch.randn_like(feat) * self.weak_sigma

        if self.p_drop_low > 0:
            drop_mask = (torch.rand(B, C, 1, 1, device=device) > self.p_drop_low).float()
        else:
            drop_mask = torch.ones(B, C, 1, 1, device=device)

        feat_low  = (feat + noise_strong) * enhance_mask * drop_mask
        feat_high = (feat + noise_weak)   * protect_mask
        residual_mask = (1.0 - enhance_mask - protect_mask).clamp_min(0.0)
        feat_aug = feat_low + feat_high + feat * residual_mask

        return feat_aug


@DETECTOR.register_module(module_name='factor_extractor')
class ExtractorDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.model = EfficientDetector(config)
        ckpt = torch.load(r'E:\DeepfakeBench\training\pretrained\ckpt_best.pth', map_location='cpu')
        # 3. 提取 state_dict
        state_dict = ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 如果之前用过 DataParallel 或 DDP
        # 4. 加载权重
        self.model.load_state_dict(state_dict, strict=False)

        self.loss_func = self.build_loss(config)
        self.config = config

        self.ffe = ForgeryUNet(in_ch=1792, base=128, mode='mask_res')
        ckpt = torch.load(r'E:\DeepfakeBench\training\pretrained\new_ffe_best.pth', map_location='cpu')
        # 3. 提取 state_dict
        state_dict = ckpt.get('model', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 如果之前用过 DataParallel 或 DDP
        # 4. 加载权重
        self.ffe.eval()
        self.ffe.load_state_dict(state_dict, strict=False)
        for p in self.ffe.parameters():
            p.requires_grad = False

        self.gate = EnergyGate()
        # __init__ 里加（一次性）
        self.register_buffer("mu_r_ema", torch.tensor(0.0))
        self.register_buffer("sig_r_ema", torch.tensor(1.0))
        self.register_buffer("mu_f_ema", torch.tensor(1.0))
        self.register_buffer("sig_f_ema", torch.tensor(1.0))
        self.ema_momentum = getattr(config, "gate_ema_m", 0.9)  # 0.9~0.99
        self.min_gap_ratio = getattr(config, "gate_min_gap", 0.05)  # 最小安全间隔（相对尺度）
        self.dropout = FakeFeatRandomDropout(drop_prob=0.2)
        self.tsne_buf = TSNEBuffer(max_per_class=4000)
        if 'random' in self.config['aug_mode']:
            self.aug = LatentAug(mode='random', use_log_energy=True,config = self.config,
                                 ema_momentum=0.95, cayley_eps=0.1, cayley_subq=32, drop_prob=0.005)
        else:
            self.aug = LatentAug(mode='energy', use_log_energy=True,config = self.config,
                                 ema_momentum=0.95, cayley_eps=0.1, cayley_subq=32, drop_prob=0.005)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config['pretrained']
        backbone = backbone_class(model_config)
        if config['pretrained'] != 'None':
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    def blend_features_batch(self, f1, f2, method='slerp'):
        """
        f1: [B1, C, H, W]
        f2: [B2, C, H, W]
        返回:
            fused: [B1*B2, C, H, W]
            alphas: [B1*B2]
        """
        B1, C, H, W = f1.shape
        B2 = f2.shape[0]
        fused_list = []
        alpha_list = []
        for i in range(B1):
            for j in range(B2):
                # 可选跳过自己与自己融合 if i==j
                if  i == j:
                    continue

                f1_sample = f1[i]
                f2_sample = f2[j]

                if method == "linear":
                    alpha = random.choice(torch.linspace(0.1, 0.9, 9)).to(f1.device)
                    fused = alpha * f1_sample + (1 - alpha) * f2_sample



                elif method == "slerp":
                    f1_flat = F.normalize(f1_sample.view(-1), dim=0)
                    f2_flat = F.normalize(f2_sample.view(-1), dim=0)
                    dot = torch.clamp(torch.dot(f1_flat, f2_flat), -1.0, 1.0)
                    omega = torch.acos(dot)
                    alpha = random.choice(torch.linspace(0.1, 0.9, 9)).to(f1.device)
                    # alpha = torch.empty((), device=f1.device).uniform_(0.95, 0.99)
                    if torch.abs(omega) < 1e-6:
                        fused_flat = f1_flat
                    else:
                        fused_flat = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * f1_flat + \
                                     (torch.sin(alpha * omega) / torch.sin(omega)) * f2_flat
                    fused = fused_flat.view(C, H, W)

                fused_list.append(fused.unsqueeze(0))
                alpha_list.append(alpha.unsqueeze(0))
        fused_batch = torch.cat(fused_list, dim=0)  # [B1*B2, C, H, W] 或 [12, C, H, W]
        alphas = torch.cat(alpha_list, dim=0)  # [B1*B2] 或 [12]
        return fused_batch, alphas
    def features(self, data_dict: dict) -> torch.tensor:
        x = self.model.features(data_dict)
        return x

        # with torch.no_grad():
        #     return self.model.features(data_dict)

    def rbf_blend(self, f1, f2, alpha, sigma=1.0):
        f1_flat = f1.view(-1)
        f2_flat = f2.view(-1)
        delta = f2_flat - f1_flat
        weight = torch.exp(- (delta ** 2).sum() / (2 * sigma ** 2))
        blended = f1_flat + alpha * weight * delta
        return blended.view_as(f1)


    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.model.classifier(features)
        # with torch.no_grad():
        #     return self.model.classifier(features)

    def soft_gate(self, energy, min_thresh, max_thresh, tau=0.5):
        lower = torch.sigmoid((energy - min_thresh) * tau)
        upper = torch.sigmoid((max_thresh - energy) * tau)
        return lower * upper  # ∈ (0,1)tat
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        device = next(self.parameters()).device

        # 分类主损
        label = data_dict['label']
        pred = pred_dict['cls']
        # if 'extra_labels' in pred_dict:
        #     label = torch.cat([label, pred_dict['extra_labels'].to(pred.device)], dim=0)
        ce_loss = self.loss_func(pred, label)



        real_factor = pred_dict.get('real_factor', None)
        fake_factor = pred_dict.get('fake_factor', None)
        syn_factor = pred_dict.get('syn_factor', None)
        fused_factor = pred_dict.get('fused_factor', None) # FFE(adapter(syn_clean))

        def rms(t):
            if t is None or t.numel() == 0: return None
            return t.flatten(1).pow(2).mean(1).sqrt()  # [B]
        device = next(self.parameters()).device
        fake_enhanced_loss = torch.tensor(0.0, device=device)
        enhanced_loss = torch.tensor(0.0, device=device)
        if 'fake_enhanced_feature' in pred_dict:
            fake_enhanced_loss = self.loss_func(pred_dict['fake_enhanced_cls'], pred_dict['fake_enhanced_labels'])


        e_r = rms(real_factor)
        e_f = rms(fake_factor)
        e_syn = rms(syn_factor)
        e_fused = rms(fused_factor)
        if 'enhanced_feature' in pred_dict:
            if (e_r is not None) and (e_f is not None):
                mu_r, sig_r = e_r.mean().detach(), e_r.std(unbiased=False).detach().clamp_min(1e-6)
                mu_f, sig_f = e_f.mean().detach(), e_f.std(unbiased=False).detach().clamp_min(1e-6)
                m = self.ema_momentum
                self.mu_r_ema = m * self.mu_r_ema + (1 - m) * mu_r
                self.sig_r_ema = m * self.sig_r_ema + (1 - m) * sig_r
                self.mu_f_ema = m * self.mu_f_ema + (1 - m) * mu_f
                self.sig_f_ema = m * self.sig_f_ema + (1 - m) * sig_f


            # 取阈值时用 EMA 版更稳
            min_thresh, max_thresh = self.gate(self.mu_r_ema, self.sig_r_ema, self.mu_f_ema, self.sig_f_ema)

            # 防交叉（给最小间隔）
            # gap_min = self.min_gap_ratio * (self.mu_r_ema + self.mu_f_ema).detach().abs().clamp_min(1e-3)
            # if min_thresh >= max_thresh:
            #     mid = 0.5 * (min_thresh + max_thresh)
            #     min_thresh = mid - gap_min
            #     max_thresh = mid + gap_min

            # mu_real = e_r.mean()
            # std_real = e_r.std()
            # mu_fake = e_f.mean()
            # std_fake = e_f.std()
            # min_thresh, max_thresh = self.gate(mu_real, std_real, mu_fake, std_fake)
            gate_weight = self.soft_gate(e_fused, min_thresh, max_thresh, tau=0.3)
            ce_per_sample = F.cross_entropy(pred_dict['enhanced_cls'], pred_dict['extra_labels'], reduction='none')
            enhanced_loss = (ce_per_sample * gate_weight).mean()

        overall = ce_loss + enhanced_loss +  self.config['enhanced_method'] * fake_enhanced_loss

        real_energy_mean = e_r.mean().detach() if e_r is not None else torch.tensor(0.0, device=device)
        fake_energy_mean = e_f.mean().detach() if e_f is not None else torch.tensor(0.0, device=device)
        syn_energy_mean = e_syn.mean().detach() if e_syn is not None else torch.tensor(0.0, device=device)
        fused_energy_mean = e_fused.mean().detach() if e_fused is not None else torch.tensor(0.0, device=device)
        return {
            'overall': overall,
            'ce': ce_loss.detach(),
            'enhanced_ce': enhanced_loss.detach(),
            'fake_enhanced_ce': fake_enhanced_loss.detach(),
            'real_energy': real_energy_mean,
            'fake_energy': fake_energy_mean,
            'syn_energy': syn_energy_mean,
            'fused_energy': fused_energy_mean,
        }

    import torch
    import torch.nn.functional as F


    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        base_N = data_dict['label'].shape[0]
        pred = pred_dict['cls'][:base_N]  # 只看原生样本
        label = data_dict['label']


        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        pred_dict = {}
        fake_enhanced_list = []
        fake_enhanced_labels_list = []
        enhanced_feature_list = []
        extra_labels = []  # 你已有，但保持一致
        features = self.features(data_dict)
        labels = data_dict['label']


        if not inference:
            mask_real = (labels == 0)
            mask_fake = (labels == 1)
            real_feat = features[mask_real]  # 可能是空张量，形状 [Br, C, H, W]
            fake_feat = features[mask_fake]

            if real_feat is not None and real_feat.numel() > 0:
                real_factor = self.ffe(real_feat)
                pred_dict['real_factor'] = real_factor.detach()

            if fake_feat is not None and fake_feat.numel() > 0:
                fake_factor = self.ffe(fake_feat)
                pred_dict['fake_factor'] = fake_factor.detach()

                # —— 生成 fake 增强（你已有的 aug 调用）——
                if self.config['fake_aug']:
                    fake_enhanced, op_idx = self.aug(fake_feat, real_feat, fake_factor=fake_factor, return_op_idx=True)
                    if fake_enhanced is not None and fake_enhanced.numel() > 0:
                        fake_enhanced_list.append(fake_enhanced)
                        fake_enhanced_labels_list.append(
                            torch.ones(fake_enhanced.size(0), dtype=torch.long, device=features.device)
                        )

                # # —— real-fake 混合生成 selected_fused（你已有）——
                if self.config['rf_blend'] and real_feat.size(0) != 0 and fake_feat.size(0) == 0:
                    rf_fused_batch, rf_alphas = self.blend_features_batch(real_feat, fake_feat)
                    mask = rf_alphas >= 0.7
                    selected_fused = rf_fused_batch[mask]
                    if selected_fused is not None and selected_fused.numel() > 0:
                        enhanced_feature_list.append(selected_fused)
                        extra_labels.append(torch.zeros(selected_fused.size(0), dtype=torch.long, device=features.device))
                        fused_factor = self.ffe(selected_fused)
                        pred_dict['fused_factor'] = fused_factor.detach()
                        del fake_factor, fused_factor
            # print(f"[aug] q1={self.aug.q1_ema.item():.4f}  q3={self.aug.q3_ema.item():.4f}")
            # if hasattr(self, 'tsne_buf'):  # 你已有 self.tsne_buf 就直接用
            #     with torch.no_grad():
            #         vis_feats = []
            #         vis_labels = []
            #
            #         if real_feat is not None and real_feat.numel() > 0:
            #             vis_feats.append(real_feat.detach())
            #             vis_labels.append(
            #                 torch.zeros(real_feat.size(0), dtype=torch.long, device=real_feat.device))  # 0
            #
            #         if fake_feat is not None and fake_feat.numel() > 0:
            #             vis_feats.append(fake_feat.detach())
            #             vis_labels.append(
            #                 torch.ones(fake_feat.size(0), dtype=torch.long, device=fake_feat.device))  # 1
            #
            #         if fake_enhanced is not None and fake_enhanced.numel() > 0:
            #             vis_feats.append(fake_enhanced.detach())
            #             vis_labels.append(torch.full((fake_enhanced.size(0),), 2, dtype=torch.long,
            #                                          device=fake_enhanced.device))  # 2
            #
            #         if len(vis_feats) > 0:
            #             vis_feats = torch.cat(vis_feats, 0)
            #             vis_labels = torch.cat(vis_labels, 0)
            #             self.tsne_buf.add(vis_feats, vis_labels)
            #             X, y = self.tsne_buf.flush_arrays()
            #     if X is not None:
            #         X_t = torch.from_numpy(X).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
            #         y_t = torch.from_numpy(y)
            #         # tsne_plot_multi(
            #         #     X_t, y_t,
            #         #     save_path=f'./debug/tsne11.2/epoch_{random.random()}.png',
            #         #     class_names={0: 'real', 1: 'fake', 2: 'fake_aug'}
            #         #     # 若细分增强类型，就写 {2:'aug_cayley',3:'aug_dropout'}
            #         # )
            #         tsne_plot_multi(X_t, y_t, "./debug/tsne11.2/tsne_pretty4.png",
            #                                class_names={0: "real", 1: "fake", 2: "fake_aug"})

        # get the prediction by classifier
        pred = self.classifier(features)
        # —— 拼接“增强的 real 混合样本” ——
        if len(enhanced_feature_list) > 0:
            pred_dict['enhanced_feature'] = torch.cat(enhanced_feature_list, 0)
            pred_dict['extra_labels'] = torch.cat(extra_labels, 0)
            enhanced_pred = self.classifier(pred_dict['enhanced_feature'])
        else:
            enhanced_pred = None

        # —— 拼接“增强的 fake 样本” ——
        if len(fake_enhanced_list) > 0:
            pred_dict['fake_enhanced_feature'] = torch.cat(fake_enhanced_list, 0)
            pred_dict['fake_enhanced_labels'] = torch.cat(fake_enhanced_labels_list, 0)
            fake_enhanced_pred = self.classifier(pred_dict['fake_enhanced_feature'])
        else:
            fake_enhanced_pred = None

        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict.update({'cls': pred, 'prob': prob,'enhanced_cls': enhanced_pred,
                           'feat': features,
                          'fake_enhanced_cls': fake_enhanced_pred,
                          })

        return pred_dict

