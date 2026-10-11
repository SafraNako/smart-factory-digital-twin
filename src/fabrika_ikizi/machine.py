"""The simulated plant: one machine (a motor-driven spindle) with a thermal state and a wear state.

Time is in hours, temperature in K or degC, power in kW, thermal capacity in kWh/K and thermal
resistance in K/kW (so the time constant ``tau = r_th * c_th`` is in hours).

* Temperature: ``c dT/dt = P - (T - T_amb) / R`` with ``P = p0 + p1 u^2 + p2 w`` (idle loss, load loss,
  friction heat that grows with wear ``w``). For a power held over a step the solution is exact:
  ``T' = a T + (1 - a)(T_amb + R P)`` with ``a = exp(-dt / tau)``.
* Wear: ``dw = kappa g(u) dt + sigma_w dW`` with ``g(u) = g0 + g1 u^2``; ``kappa`` is a severity
  factor (1 = nominal). The machine fails when ``w`` reaches 1, a first passage of a Wiener process.
* Vibration (the quantity a sensor sees) is ``v = v0 + kv w``.

Faults change the plant, not the sensors: ``bearing`` multiplies ``kappa`` from ``start`` on, and
``cooling`` multiplies the thermal resistance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MachineParams:
    t_amb: float = 22.0        # degC
    r_th: float = 8.0          # K/kW
    c_th: float = 0.0417       # kWh/K  (tau = 0.33 h)
    p0: float = 0.4            # kW idle loss
    p1: float = 3.2            # kW extra loss at full load
    p2: float = 2.5            # kW extra loss at wear 1
    v0: float = 1.2            # mm/s healthy vibration
    kv: float = 4.5            # mm/s per unit wear
    g0: float = 7e-4           # wear per hour at idle
    g1: float = 1.5e-3         # wear per hour at full load (times u^2)
    sigma_w: float = 0.004     # wear diffusion, per sqrt(hour)
    q_t: float = 0.05          # K, unmodelled temperature disturbance per step (std)
    w0: float = 0.05           # initial wear

    @property
    def tau(self):
        return self.r_th * self.c_th

    def wear_rate(self, u):
        """Nominal wear per hour at load fraction ``u``."""
        u = np.asarray(u, dtype=float)
        return self.g0 + self.g1 * u * u

    def heat_power(self, u, w):
        u = np.asarray(u, dtype=float)
        return self.p0 + self.p1 * u * u + self.p2 * np.asarray(w, dtype=float)


@dataclass(frozen=True)
class Fault:
    kind: str                  # "bearing" or "cooling"
    start: float               # hours
    magnitude: float = 3.0     # factor applied to kappa (bearing) or to r_th (cooling)

    def __post_init__(self):
        if self.kind not in ("bearing", "cooling"):
            raise ValueError("fault kind must be 'bearing' or 'cooling'")
        if self.magnitude <= 0:
            raise ValueError("fault magnitude must be positive")


def daily_load(t):
    """A three-shift day: 0.85 for 8 h, 0.55 for 8 h, 0.10 for 8 h (a quiet night shift)."""
    h = np.mod(np.asarray(t, dtype=float), 24.0)
    return np.where(h < 8.0, 0.85, np.where(h < 16.0, 0.55, 0.10))


def mean_wear_rate(params: MachineParams, load=daily_load, horizon=24.0, n=2400):
    """Average nominal wear rate over a planning horizon: what the twin assumes for the future."""
    t = np.linspace(0.0, horizon, n, endpoint=False)
    return float(np.mean(params.wear_rate(load(t))))


@dataclass
class Truth:
    t: np.ndarray
    u: np.ndarray
    temp: np.ndarray
    wear: np.ndarray
    kappa: np.ndarray          # the severity actually applied at each step
    r_mult: np.ndarray
    dt: float
    fail_index: int            # first step with wear >= 1, or -1 if it never fails in the window
    params: MachineParams = field(repr=False, default=None)

    @property
    def vibration(self):
        return self.params.v0 + self.params.kv * self.wear

    @property
    def fail_time(self):
        return float(self.t[self.fail_index]) if self.fail_index >= 0 else float("nan")


def simulate_plant(params: MachineParams, hours, dt=0.25, seed=0, faults=(), load=daily_load, kappa=1.0,
                   stop_at_failure=False):
    """Simulate the true state. ``kappa`` is the machine's own wear severity (1 = nominal)."""
    rng = np.random.default_rng(seed)
    n = int(round(hours / dt)) + 1
    t = np.arange(n) * dt
    u = np.asarray(load(t), dtype=float)
    kmult = np.full(n, float(kappa))
    rmult = np.ones(n)
    for f in faults:
        idx = t >= f.start
        if f.kind == "bearing":
            kmult[idx] *= f.magnitude
        else:
            rmult[idx] *= f.magnitude
    inc = kmult[:-1] * params.wear_rate(u[:-1]) * dt + params.sigma_w * np.sqrt(dt) * rng.standard_normal(n - 1)
    wear = params.w0 + np.concatenate([[0.0], np.cumsum(inc)])
    a = np.exp(-dt / params.tau)
    shocks = params.q_t * rng.standard_normal(n - 1)
    temp = np.empty(n)
    temp[0] = params.t_amb + params.r_th * float(params.heat_power(u[0], wear[0]))
    for k in range(n - 1):
        t_inf = params.t_amb + params.r_th * rmult[k] * (params.p0 + params.p1 * u[k] ** 2 + params.p2 * wear[k])
        temp[k + 1] = a * temp[k] + (1.0 - a) * t_inf + shocks[k]
    crossed = np.flatnonzero(wear >= 1.0)
    fail = int(crossed[0]) if len(crossed) else -1
    if stop_at_failure and fail >= 0:
        sl = slice(0, fail + 1)
        return Truth(t[sl], u[sl], temp[sl], wear[sl], kmult[sl], rmult[sl], dt, fail, params)
    return Truth(t, u, temp, wear, kmult, rmult, dt, fail, params)
