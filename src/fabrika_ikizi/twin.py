"""The machine twin: a Kalman filter over ``(temperature, wear, severity)`` fed by the sensor stream.

The state is ``x = [T, w, kappa]``. Given the load ``u`` held over a step of ``dt`` hours::

    T'     = a T + (1 - a) (T_amb + R (p0 + p1 u^2)) + (1 - a) R p2 w + noise
    w'     = w + kappa g(u) dt + noise
    kappa' = kappa + noise                      (a slowly drifting wear-rate multiplier)

with ``a = exp(-dt / tau)``, observed through the temperature and the vibration sensor. The standardised
innovations feed two CUSUM charts; a rising ``kappa`` estimate is the wear-acceleration alarm; the
posterior of ``(w, kappa)`` gives the distribution of the remaining life.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import monitor, rul
from .kalman import KalmanFilter, nis
from .machine import MachineParams, mean_wear_rate
from .sensors import SensorSpec


@dataclass
class TwinRun:
    t: np.ndarray
    x: np.ndarray                 # (n, 3) posterior means
    P: np.ndarray                 # (n, 3, 3) posterior covariances
    z: np.ndarray                 # (n, 2) standardised innovations (NaN where the sample is missing)
    nis: np.ndarray               # (n,)
    alarms: list = field(default_factory=list)      # (time, kind)
    twin: object = None
    active: np.ndarray = None                       # (n,) True while a chart statistic is high (sensor-model disagreement)

    @property
    def temp(self):
        return self.x[:, 0]

    @property
    def wear(self):
        return self.x[:, 1]

    @property
    def kappa(self):
        return self.x[:, 2]

    @property
    def wear_std(self):
        return np.sqrt(self.P[:, 1, 1])

    def rul(self, index, size=4000, seed=0, propagate=True, level=0.8):
        """Median and central interval of the remaining life (hours) from the estimate at ``index``."""
        twin = self.twin
        s = rul.rul_samples(self.x[index, 1], self.x[index, 2], self.P[index][1:, 1:], twin.g_future,
                            twin.params.sigma_w, size=size, rng=np.random.default_rng(seed), propagate=propagate)
        lo, hi = rul.interval(s, level)
        return float(np.median(s)), lo, hi


class MachineTwin:
    def __init__(self, params: MachineParams, dt, temp_sensor: SensorSpec, vib_sensor: SensorSpec, q_kappa=0.03,
                 g_future=None, arl0=10000.0, k=0.5, kappa_alarm=1.8, prior_wear=0.1, prior_wear_sd=0.2,
                 prior_kappa_sd=0.5, model_params: MachineParams = None):
        """``model_params`` lets the twin use parameters that differ from the plant's (model mismatch)."""
        self.params = model_params or params
        self.dt = dt
        self.q_kappa = q_kappa
        self.a = float(np.exp(-dt / self.params.tau))
        self.R = np.diag([temp_sensor.effective_sigma() ** 2, vib_sensor.effective_sigma() ** 2])
        self.g_future = g_future if g_future is not None else mean_wear_rate(self.params)
        self.h = monitor.cusum_threshold(arl0, k)
        self.k = k
        self.kappa_alarm = kappa_alarm
        self.prior_wear, self.prior_wear_sd, self.prior_kappa_sd = prior_wear, prior_wear_sd, prior_kappa_sd

    # model matrices -----------------------------------------------------------------------
    def F(self, u):
        p, a = self.params, self.a
        return np.array([[a, (1.0 - a) * p.r_th * p.p2, 0.0],
                         [0.0, 1.0, float(p.wear_rate(u)) * self.dt],
                         [0.0, 0.0, 1.0]])

    def b(self, u):
        p, a = self.params, self.a
        return np.array([(1.0 - a) * (p.t_amb + p.r_th * (p.p0 + p.p1 * u * u)), 0.0, 0.0])

    def Q(self):
        p = self.params
        return np.diag([p.q_t ** 2, p.sigma_w ** 2 * self.dt, self.q_kappa ** 2 * self.dt])

    def H(self):
        return np.array([[1.0, 0.0, 0.0], [0.0, self.params.kv, 0.0]])

    def c(self):
        return np.array([0.0, self.params.v0])

    def initial(self, y0, u0):
        p = self.params
        t0 = y0[0] if not np.isnan(y0[0]) else p.t_amb + p.r_th * (p.p0 + p.p1 * u0 * u0 + p.p2 * self.prior_wear)
        x0 = np.array([t0, self.prior_wear, 1.0])
        P0 = np.diag([4.0 ** 2, self.prior_wear_sd ** 2, self.prior_kappa_sd ** 2])
        return x0, P0

    # streaming -----------------------------------------------------------------------------
    def run(self, t, u, y_temp, y_vib):
        """Process a whole stream (one entry per sample) and return a :class:`TwinRun`."""
        t = np.asarray(t, dtype=float)
        u = np.asarray(u, dtype=float)
        y = np.column_stack([y_temp, y_vib]).astype(float)
        n = len(t)
        kf = KalmanFilter(*self.initial(y[0], u[0]))
        H, c, Rm, Q = self.H(), self.c(), self.R, self.Q()
        xs, Ps = np.empty((n, 3)), np.empty((n, 3, 3))
        zs, nises = np.full((n, 2), np.nan), np.full(n, np.nan)
        active = np.zeros(n, dtype=bool)
        charts = [monitor.Cusum(self.k, self.h) for _ in range(2)]
        alarms, kappa_on = [], False
        for i in range(n):
            nu, S = kf.update(y[i], H, c, Rm)
            ok = ~np.isnan(nu)
            zs[i, ok] = nu[ok] / np.sqrt(np.diag(S)[ok])
            nises[i] = nis(nu, S)[0]
            xs[i], Ps[i] = kf.x, kf.P
            for ch, name in enumerate(("T", "v")):
                hit = charts[ch].update(float(zs[i, ch]))
                if hit:
                    alarms.append((float(t[i]), f"{name}{'+' if hit > 0 else '-'}"))
            active[i] = charts[0].active or charts[1].active
            if not kappa_on and kf.x[2] > self.kappa_alarm:
                alarms.append((float(t[i]), "wear-rate"))
                kappa_on = True
            elif kappa_on and kf.x[2] < self.kappa_alarm - 0.2:
                kappa_on = False
            if i < n - 1:
                kf.predict(self.F(u[i]), self.b(u[i]), Q)
        return TwinRun(t, xs, Ps, zs, nises, alarms, self, active)
