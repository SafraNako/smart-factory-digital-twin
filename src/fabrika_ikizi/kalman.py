"""A linear Kalman filter that skips missing (NaN) measurements, and an information-form batch solution.

The batch solver stacks the whole trajectory into one weighted least-squares problem and solves the
normal equations; for a linear-Gaussian model it must reproduce the filter's final estimate and
covariance, which is an independent check of the filter's algebra.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    """``x' = F x + b + w, w ~ N(0, Q);  y = H x + c + v, v ~ N(0, R)`` with a Joseph-form update."""

    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self, F, b, Q):
        self.x = F @ self.x + b
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, y, H, c, R):
        """Update with the available entries of ``y``; returns ``(innovation, S)`` with NaN where missing.

        ``S`` is the full innovation covariance of the available entries, embedded in a matrix of
        NaN rows/columns for the missing ones, so that ``sqrt(diag(S))`` standardises each channel.
        """
        y = np.asarray(y, dtype=float)
        m = len(y)
        have = ~np.isnan(y)
        nu_full = np.full(m, np.nan)
        s_full = np.full((m, m), np.nan)
        if not have.any():
            return nu_full, s_full
        Hm, Rm = H[have], R[np.ix_(have, have)]
        S = Hm @ self.P @ Hm.T + Rm
        nu = y[have] - (Hm @ self.x + c[have])
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(len(self.x)) - K @ Hm
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        nu_full[have] = nu
        s_full[np.ix_(have, have)] = S
        return nu_full, s_full


def nis(nu, S):
    """Normalised innovation squared over the available channels (chi-square with that many degrees of freedom)."""
    have = ~np.isnan(nu)
    if not have.any():
        return float("nan"), 0
    n = nu[have]
    return float(n @ np.linalg.solve(S[np.ix_(have, have)], n)), int(have.sum())


def nees(error, P):
    """Normalised estimation error squared: chi-square with ``len(error)`` degrees of freedom if the filter is consistent."""
    return float(error @ np.linalg.solve(P, error))


def scalar_steady_state(q, r):
    """Steady-state prior variance of a scalar random walk observed in noise: ``P = P - P^2/(P + r) + q``.

    Solves ``P^2 - q P - q r = 0`` for the positive root (closed form).
    """
    return 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))


def batch_estimate(x0, P0, Fs, bs, Qs, ys, Hs, cs, Rs):
    """Weighted least squares over the whole window.

    Unknowns are ``x_0 .. x_N`` stacked. Residual blocks: the prior on ``x_0`` (precision ``P0^-1``),
    the dynamics ``x_{k+1} - F_k x_k - b_k`` (precision ``Q_k^-1``) and the measurements ``y_k - H x_k - c``
    (precision ``R^-1``, NaN entries dropped). ``ys`` has ``N + 1`` entries (one per state), ``Fs/bs/Qs`` have ``N``.
    Returns the estimate of ``x_N`` and its covariance (the last diagonal block of the inverse information).
    """
    n = len(x0)
    N = len(Fs)
    dim = n * (N + 1)
    info = np.zeros((dim, dim))
    rhs = np.zeros(dim)

    def block(k):
        return slice(k * n, (k + 1) * n)

    P0i = np.linalg.inv(P0)
    info[block(0), block(0)] += P0i
    rhs[block(0)] += P0i @ x0
    for k in range(N):
        Qi = np.linalg.inv(Qs[k])
        F, b = Fs[k], bs[k]
        # residual r = x_{k+1} - F x_k - b = [-F, I] [x_k; x_{k+1}] - b
        A = np.hstack([-F, np.eye(n)])
        sl = np.r_[np.arange(k * n, (k + 2) * n)]
        info[np.ix_(sl, sl)] += A.T @ Qi @ A
        rhs[sl] += A.T @ Qi @ b
    for k in range(N + 1):
        y = np.asarray(ys[k], dtype=float)
        have = ~np.isnan(y)
        if not have.any():
            continue
        Hm = Hs[k][have]
        Ri = np.linalg.inv(Rs[k][np.ix_(have, have)])
        info[block(k), block(k)] += Hm.T @ Ri @ Hm
        rhs[block(k)] += Hm.T @ Ri @ (y[have] - cs[k][have])
    cov = np.linalg.inv(info)
    est = cov @ rhs
    return est[block(N)], cov[block(N), block(N)]
