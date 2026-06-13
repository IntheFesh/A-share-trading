"""高斯隐马尔可夫区制模型(完整 Baum-Welch + 因果滤波后验)。Phase 3。对应 v3.1 §5.3(a)。

L0 regime 的第二条可选实现:对市场收益建高斯 HMM,输出**状态后验概率**(非 0/1 硬标签)。
v3.1 诚实纪律:regime 识别天然滞后——**滤波(filtered,只用历史)可实盘;平滑(smoothed,用了未来)
严禁进回测**。本实现两者都给,但 ``filtered_posterior`` 是唯一可上线的;``smoothed_posterior`` 仅供研究,
显式标注前视。参数用 Baum-Welch 在训练窗内估计(滚动重估由 run_train 负责)。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _gaussian_logpdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


class GaussianHMM:
    """K 状态一维高斯 HMM。log 空间前后向,数值稳定。"""

    def __init__(self, n_states: int = 2, n_iter: int = 50, tol: float = 1e-4,
                 var_floor: float = 1e-8, seed: int = 0) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.var_floor = var_floor
        self.rng = np.random.default_rng(seed)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([_gaussian_logpdf(x, self.means_[k], self.vars_[k])
                                for k in range(self.n_states)])

    def _init_params(self, x: np.ndarray) -> None:
        K = self.n_states
        qs = np.quantile(x, np.linspace(0.1, 0.9, K))
        self.means_ = qs.astype("float64")
        self.vars_ = np.full(K, max(float(np.var(x)), self.var_floor))
        self.startprob_ = np.full(K, 1.0 / K)
        self.transmat_ = np.full((K, K), 0.1 / max(1, K - 1))
        np.fill_diagonal(self.transmat_, 0.9)

    def _forward(self, logB: np.ndarray):
        T, K = logB.shape
        log_alpha = np.empty((T, K))
        log_start, log_trans = np.log(self.startprob_ + 1e-300), np.log(self.transmat_ + 1e-300)
        log_alpha[0] = log_start + logB[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_trans[:, k]) + logB[t, k]
        return log_alpha, log_trans

    def fit(self, x: "np.ndarray") -> "GaussianHMM":
        x = np.asarray(x, dtype="float64")
        x = x[~np.isnan(x)]
        if len(x) < self.n_states * 2:
            raise ValueError("样本过少,无法拟合 HMM")
        self._init_params(x)
        T, K = len(x), self.n_states
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            logB = self._log_emission(x)
            log_alpha, log_trans = self._forward(logB)
            ll = logsumexp(log_alpha[-1])
            # 后向
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                for k in range(K):
                    log_beta[t, k] = logsumexp(log_trans[k, :] + logB[t + 1] + log_beta[t + 1])
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            # xi 累加
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                m = (log_alpha[t][:, None] + log_trans + logB[t + 1][None, :]
                     + log_beta[t + 1][None, :] - ll)
                xi_sum += np.exp(m)
            # M 步
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi_sum / xi_sum.sum(axis=1, keepdims=True).clip(1e-300)
            gsum = gamma.sum(axis=0).clip(1e-300)
            self.means_ = (gamma * x[:, None]).sum(axis=0) / gsum
            self.vars_ = ((gamma * (x[:, None] - self.means_[None, :]) ** 2).sum(axis=0) / gsum
                          ).clip(self.var_floor)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def filtered_posterior(self, x: "np.ndarray") -> np.ndarray:
        """因果滤波后验 P(s_t | x_1..t)(只用历史,可实盘)。返回 (T, K)。"""
        x = np.asarray(x, dtype="float64")
        logB = self._log_emission(x)
        log_alpha, _ = self._forward(logB)
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def smoothed_posterior(self, x: "np.ndarray") -> np.ndarray:
        """平滑后验 P(s_t | x_1..T)(**用了未来,前视,严禁进回测**),仅供研究。"""
        x = np.asarray(x, dtype="float64")
        logB = self._log_emission(x)
        log_alpha, log_trans = self._forward(logB)
        T, K = logB.shape
        ll = logsumexp(log_alpha[-1])
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(log_trans[k, :] + logB[t + 1] + log_beta[t + 1])
        return np.exp(log_alpha + log_beta - ll)


def compute_regime_state_probs(returns: "np.ndarray", *, n_states: int = 2, seed: int = 0):
    """拟合 HMM 并返回因果滤波的状态后验(供 L0 以连续概率参与 m_t / 交互项)。

    返回 (probs[T,K], 状态按均值升序的索引)。bear=均值最低的状态。价格层:用 adj 收益。
    """
    r = np.asarray(returns, dtype="float64")
    model = GaussianHMM(n_states=n_states, seed=seed).fit(r)
    probs = model.filtered_posterior(r)
    order = np.argsort(model.means_)  # 低均值=熊态
    return probs, order
