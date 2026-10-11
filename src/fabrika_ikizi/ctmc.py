"""Exact solution of a serial line with exponential service, failures and repairs, finite buffers.

The state is the waiting-room level (Poisson source only), the buffer levels and, for every station,
whether it is up and whether it is empty, processing or holding a finished part. The transition rules are
those of :mod:`line` written again as a pure function on states, and the stationary distribution comes
from a linear solve of the balance equations, so it checks the event simulation without any sampling.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .line import DONE, EMPTY, PROC, Line


@dataclass
class CtmcResult:
    n_states: int
    throughput: float                 # parts per minute
    busy: np.ndarray
    blocked: np.ndarray
    starved: np.ndarray
    down: np.ndarray
    mean_buffers: np.ndarray
    mean_queue: float
    wip: float
    loss_probability: float           # chance an arrival finds the waiting room full (Poisson source), else 0


def check_supported(line: Line):
    for s in line.stations:
        if s.dist != "exp":
            raise ValueError("the exact solver needs exponential service times")
        if s.scrap:
            raise ValueError("the exact solver does not model scrap")
    if any(math.isinf(b) for b in line.buffers):
        raise ValueError("the exact solver needs finite buffers")
    if line.arrival_rate is not None and math.isinf(line.queue_capacity):
        raise ValueError("the exact solver needs a finite waiting room for a Poisson source")


def _settle(line: Line, q, bufs, ups, sts):
    """Apply the instantaneous moves until nothing changes (lists are modified in place)."""
    K = len(line.stations)
    cap = line.buffers
    saturated = line.arrival_rate is None
    changed = True
    while changed:
        changed = False
        for i in range(K - 1, -1, -1):
            if sts[i] == DONE:
                if i == K - 1:
                    sts[i] = EMPTY
                    changed = True
                elif bufs[i] < cap[i]:
                    bufs[i] += 1
                    sts[i] = EMPTY
                    changed = True
                elif cap[i] == 0 and sts[i + 1] == EMPTY:
                    sts[i + 1] = PROC
                    sts[i] = EMPTY
                    changed = True
            if sts[i] == EMPTY:
                if i == 0:
                    if saturated:
                        sts[0] = PROC
                        changed = True
                    elif q[0] > 0:
                        q[0] -= 1
                        sts[0] = PROC
                        changed = True
                elif bufs[i - 1] > 0:
                    bufs[i - 1] -= 1
                    sts[i] = PROC
                    changed = True


def solve(line: Line) -> CtmcResult:
    check_supported(line)
    K = len(line.stations)
    st = line.stations
    lam = line.arrival_rate
    Q0 = None if lam is None else int(line.queue_capacity)

    def pack(q, bufs, ups, sts):
        return (q[0], tuple(bufs), tuple(ups), tuple(sts))

    def unpack(state):
        return [state[0]], list(state[1]), list(state[2]), list(state[3])

    start_q, start_bufs, start_ups, start_sts = [0], [0] * (K - 1), [1] * K, [EMPTY] * K
    _settle(line, start_q, start_bufs, start_ups, start_sts)
    start = pack(start_q, start_bufs, start_ups, start_sts)
    index = {start: 0}
    order = [start]
    edges = []                                   # (from, to, rate)
    queue = deque([start])

    def emit(src, new_state, rate):
        if new_state not in index:
            index[new_state] = len(order)
            order.append(new_state)
            queue.append(new_state)
        edges.append((index[src], index[new_state], rate))

    while queue:
        s = queue.popleft()
        if lam is not None and s[0] < Q0:
            q, bufs, ups, sts = unpack(s)
            q[0] += 1
            _settle(line, q, bufs, ups, sts)
            emit(s, pack(q, bufs, ups, sts), lam)
        for i in range(K):
            if s[2][i] and s[3][i] == PROC:
                q, bufs, ups, sts = unpack(s)
                sts[i] = DONE
                _settle(line, q, bufs, ups, sts)
                emit(s, pack(q, bufs, ups, sts), 1.0 / st[i].mean_service)
            if math.isfinite(st[i].mtbf):
                q, bufs, ups, sts = unpack(s)
                if s[2][i]:
                    ups[i] = 0
                    emit(s, pack(q, bufs, ups, sts), 1.0 / st[i].mtbf)
                else:
                    ups[i] = 1
                    emit(s, pack(q, bufs, ups, sts), 1.0 / st[i].mttr)

    n = len(order)
    G = np.zeros((n, n))
    for a, b, r in edges:
        if a != b:
            G[a, b] += r
    np.fill_diagonal(G, -G.sum(axis=1))
    A = G.T.copy()
    A[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    pi = np.linalg.solve(A, rhs)

    busy, blocked, starved, down = (np.zeros(K) for _ in range(4))
    mean_buf = np.zeros(max(K - 1, 0))
    mean_q = wip = loss = 0.0
    for p, s in zip(pi, order):
        for i in range(K):
            if not s[2][i]:
                down[i] += p
            elif s[3][i] == PROC:
                busy[i] += p
            elif s[3][i] == DONE:
                blocked[i] += p
            else:
                starved[i] += p
        for j in range(K - 1):
            mean_buf[j] += p * s[1][j]
        mean_q += p * s[0]
        wip += p * (s[0] + sum(s[1]) + sum(1 for x in s[3] if x != EMPTY))
        if lam is not None and s[0] >= Q0:
            loss += p
    throughput = float(busy[K - 1] / st[K - 1].mean_service)
    return CtmcResult(n, throughput, busy, blocked, starved, down, mean_buf, mean_q, wip, loss)


def two_station_closed_form(mu1, mu2, buffer):
    """Throughput of two reliable exponential stations (rates ``mu1``, ``mu2``) with a buffer of ``buffer`` places.

    The parts held in the buffer, in station 2 and (blocked) in station 1 form a birth-death chain on
    ``0 .. buffer + 2`` with birth rate ``mu1`` and death rate ``mu2``; throughput is ``mu2 (1 - pi_0)``.
    """
    rho = mu1 / mu2
    n = buffer + 3
    weights = np.array([rho ** k for k in range(n)])
    pi0 = weights[0] / weights.sum()
    return mu2 * (1.0 - pi0)


def mm1k(lam, mu, k):
    """M/M/1/K (system capacity ``k``): loss probability and mean number in system."""
    rho = lam / mu
    w = np.array([rho ** n for n in range(k + 1)])
    pi = w / w.sum()
    return float(pi[-1]), float((np.arange(k + 1) * pi).sum())
