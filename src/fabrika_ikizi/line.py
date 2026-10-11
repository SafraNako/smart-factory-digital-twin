"""Discrete-event simulation of a serial production line.

Stations in series with finite (or infinite) buffers between them. A part finished at a station moves to
the next buffer if there is room (a zero-size buffer hands it straight to an empty next station) and
otherwise stays where it is, blocking the station (blocking after service). A station loads a part as soon
as it is empty and one is available; processing only advances while the station is up (failures come
from the calendar: exponential up and repair times, a part in process resumes after the repair).
The first station is either saturated (never starves) or fed by a Poisson stream into a waiting room.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

EMPTY, PROC, DONE = 0, 1, 2


@dataclass(frozen=True)
class Station:
    name: str
    mean_service: float             # minutes
    dist: str = "exp"               # exp | det | erlang:k | lognorm:cv
    mtbf: float = math.inf          # minutes between failures (calendar time)
    mttr: float = 0.0               # minutes to repair
    scrap: float = 0.0              # probability that a finished part is scrapped

    def __post_init__(self):
        if self.mean_service <= 0:
            raise ValueError("mean_service must be positive")
        if not 0.0 <= self.scrap < 1.0:
            raise ValueError("scrap must be in [0, 1)")
        if math.isfinite(self.mtbf) and (self.mtbf <= 0 or self.mttr <= 0):
            raise ValueError("a failing station needs positive mtbf and mttr")
        sampler(self.dist, self.mean_service)       # validates the distribution name

    @property
    def availability(self):
        return self.mtbf / (self.mtbf + self.mttr) if math.isfinite(self.mtbf) else 1.0

    @property
    def capacity(self):
        """Parts per minute the station can deliver when it never starves or blocks."""
        return self.availability / self.mean_service


@dataclass(frozen=True)
class Line:
    stations: tuple
    buffers: tuple = ()                         # capacities between consecutive stations
    arrival_rate: float = None                  # parts per minute; None = saturated source
    queue_capacity: float = math.inf            # waiting room before the first station (Poisson source)

    def __post_init__(self):
        if len(self.stations) < 1:
            raise ValueError("a line needs at least one station")
        if len(self.buffers) != len(self.stations) - 1:
            raise ValueError("buffers must have one entry per gap between stations")
        if any(b < 0 or (math.isfinite(b) and b != int(b)) for b in self.buffers):
            raise ValueError("buffer capacities must be non-negative integers or infinity")
        if self.arrival_rate is not None and self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")

    @property
    def bottleneck(self):
        """Index of the station with the smallest standalone capacity ``availability / mean_service``."""
        return int(np.argmin([s.capacity for s in self.stations]))

    def with_station(self, i, **changes):
        st = list(self.stations)
        st[i] = replace(st[i], **changes)
        return replace(self, stations=tuple(st))

    def with_buffer(self, i, capacity):
        b = list(self.buffers)
        b[i] = capacity
        return replace(self, buffers=tuple(b))


def sampler(dist, mean):
    """A function ``rng -> service time`` with the given mean."""
    if dist == "exp":
        return lambda rng: rng.exponential(mean)
    if dist == "det":
        return lambda rng: mean
    if dist.startswith("erlang:"):
        k = int(dist.split(":")[1])
        if k < 1:
            raise ValueError("erlang shape must be >= 1")
        return lambda rng: rng.gamma(k, mean / k)
    if dist.startswith("lognorm:"):
        cv = float(dist.split(":")[1])
        if cv <= 0:
            raise ValueError("lognormal cv must be positive")
        s2 = math.log(1.0 + cv * cv)
        mu = math.log(mean) - 0.5 * s2
        return lambda rng: rng.lognormal(mu, math.sqrt(s2))
    raise ValueError(f"unknown service distribution {dist!r}")


@dataclass
class LineResult:
    horizon: float
    warmup: float
    throughput: float                 # good parts per minute (window)
    exits: int
    scrapped: int
    lost: int
    busy: np.ndarray                  # per station, fraction of the window
    blocked: np.ndarray
    starved: np.ndarray
    down: np.ndarray
    mean_buffers: np.ndarray
    mean_queue: float
    wip: float                        # mean number of parts in the system
    cycle_time: float                 # mean time in the system of departing parts
    mean_wait: float                  # mean wait before the first station
    log: dict = field(default_factory=dict)

    @property
    def little_residual(self):
        """Relative gap in Little's law: ``(L - lambda W) / L`` with ``lambda`` the departure rate."""
        window = self.horizon - self.warmup
        rate = (self.exits + self.scrapped) / window
        return (self.wip - rate * self.cycle_time) / self.wip if self.wip > 0 else 0.0

    def oee(self, stations):
        """Per station ``(availability, performance, quality, oee)`` from the window fractions.

        Availability is the up fraction, performance the share of up time spent processing, quality the
        non-scrap share; their product is the busy fraction times quality.
        """
        out = []
        for i, st in enumerate(stations):
            avail = 1.0 - self.down[i]
            perf = self.busy[i] / avail if avail > 0 else 0.0
            quality = 1.0 - st.scrap
            out.append((avail, perf, quality, avail * perf * quality))
        return out


def simulate(line: Line, horizon, warmup=0.0, seed=0):
    rng = np.random.default_rng(seed)
    K = len(line.stations)
    stations = line.stations
    draw = [sampler(s.dist, s.mean_service) for s in stations]
    cap = list(line.buffers)
    saturated = line.arrival_rate is None
    if horizon <= warmup:
        raise ValueError("horizon must exceed warmup")

    status = [EMPTY] * K
    up = [True] * K
    entry = [0.0] * K
    svc = [0.0] * K
    remaining = [0.0] * K
    comp_time = [0.0] * K
    epoch = [0] * K
    buf = [deque() for _ in range(K - 1)]
    queue = deque()
    heap, seq = [], 0

    def push(t, kind, i, ep=0):
        nonlocal seq
        seq += 1
        heapq.heappush(heap, (t, seq, kind, i, ep))

    t_last = 0.0
    window = horizon - warmup
    acc_state = np.zeros((4, K))                   # busy, blocked, starved, down
    acc_buf = np.zeros(max(K - 1, 0))
    acc_queue = acc_wip = 0.0
    stats = dict(exits=0, scrapped=0, lost=0, soj=0.0, n_soj=0, wait=0.0, n_wait=0)
    log = {"service": [[] for _ in range(K)], "scrap": np.zeros(K, dtype=int), "up_time": np.zeros(K),
           "n_fail": np.zeros(K, dtype=int),
           "down_time": np.zeros(K), "n_repair": np.zeros(K, dtype=int)}
    up_since = [0.0] * K
    down_since = [0.0] * K

    def accumulate(t_new):
        nonlocal t_last, acc_queue, acc_wip
        lo, hi = max(t_last, warmup), min(t_new, horizon)
        if hi > lo:
            d = hi - lo
            for i in range(K):
                if not up[i]:
                    acc_state[3, i] += d
                elif status[i] == PROC:
                    acc_state[0, i] += d
                elif status[i] == DONE:
                    acc_state[1, i] += d
                else:
                    acc_state[2, i] += d
            for j in range(K - 1):
                acc_buf[j] += d * len(buf[j])
            acc_queue += d * len(queue)
            acc_wip += d * (len(queue) + sum(len(b) for b in buf) + sum(1 for s in status if s != EMPTY))
        t_last = t_new

    def depart(t, born, scrapped):
        if t >= warmup:
            stats["scrapped" if scrapped else "exits"] += 1
            stats["soj"] += t - born
            stats["n_soj"] += 1

    def load(i, born, t, waited=None):
        status[i] = PROC
        entry[i] = born
        svc[i] = draw[i](rng)
        if i == 0 and waited is not None and t >= warmup:
            stats["wait"] += waited
            stats["n_wait"] += 1
        if up[i]:
            comp_time[i] = t + svc[i]
            push(comp_time[i], 0, i, epoch[i])
        else:
            remaining[i] = svc[i]

    def settle(t):
        changed = True
        while changed:
            changed = False
            for i in range(K - 1, -1, -1):
                if status[i] == DONE:
                    if i == K - 1:
                        depart(t, entry[i], False)
                        status[i] = EMPTY
                        changed = True
                    elif len(buf[i]) < cap[i]:
                        buf[i].append(entry[i])
                        status[i] = EMPTY
                        changed = True
                    elif cap[i] == 0 and status[i + 1] == EMPTY:
                        load(i + 1, entry[i], t)
                        status[i] = EMPTY
                        changed = True
                if status[i] == EMPTY:
                    if i == 0:
                        if saturated:
                            load(0, t, t)
                            changed = True
                        elif queue:
                            born = queue.popleft()
                            load(0, born, t, waited=t - born)
                            changed = True
                    elif buf[i - 1]:
                        load(i, buf[i - 1].popleft(), t)
                        changed = True

    for i, s in enumerate(stations):
        if math.isfinite(s.mtbf):
            push(rng.exponential(s.mtbf), 1, i)
    if not saturated:
        push(rng.exponential(1.0 / line.arrival_rate), 3, 0)
    settle(0.0)

    while heap:
        t, _, kind, i, ep = heapq.heappop(heap)
        if t > horizon:
            break
        accumulate(t)
        if kind == 0:                                           # completion
            if ep != epoch[i]:
                continue
            log["service"][i].append(svc[i])
            if stations[i].scrap > 0 and rng.random() < stations[i].scrap:
                log["scrap"][i] += 1
                depart(t, entry[i], True)
                status[i] = EMPTY
            else:
                status[i] = DONE
            settle(t)
        elif kind == 1:                                         # failure
            up[i] = False
            log["up_time"][i] += t - up_since[i]
            log["n_fail"][i] += 1
            if status[i] == PROC:
                remaining[i] = comp_time[i] - t
                epoch[i] += 1
            down_since[i] = t
            push(t + rng.exponential(stations[i].mttr), 2, i)
        elif kind == 2:                                         # repair
            up[i] = True
            log["down_time"][i] += t - down_since[i]
            log["n_repair"][i] += 1
            up_since[i] = t
            if status[i] == PROC:
                comp_time[i] = t + remaining[i]
                push(comp_time[i], 0, i, epoch[i])
            push(t + rng.exponential(stations[i].mtbf), 1, i)
        else:                                                   # arrival
            if len(queue) < line.queue_capacity:
                queue.append(t)
                settle(t)
            elif t >= warmup:
                stats["lost"] += 1
            push(t + rng.exponential(1.0 / line.arrival_rate), 3, 0)
    accumulate(horizon)
    # exposures up to the horizon (right-censored last periods)
    for i in range(K):
        if math.isfinite(stations[i].mtbf):
            if up[i]:
                log["up_time"][i] += horizon - up_since[i]
            else:
                log["down_time"][i] += horizon - down_since[i]
    soj = stats["soj"] / stats["n_soj"] if stats["n_soj"] else float("nan")
    wait = stats["wait"] / stats["n_wait"] if stats["n_wait"] else 0.0
    return LineResult(
        horizon=horizon, warmup=warmup, throughput=stats["exits"] / window, exits=stats["exits"],
        scrapped=stats["scrapped"], lost=stats["lost"], busy=acc_state[0] / window,
        blocked=acc_state[1] / window, starved=acc_state[2] / window, down=acc_state[3] / window,
        mean_buffers=acc_buf / window, mean_queue=acc_queue / window, wip=acc_wip / window,
        cycle_time=soj, mean_wait=wait, log=log)


def replicate(line: Line, horizon, replications, warmup=0.0, seed=0):
    """Independent replications; returns the list of results."""
    return [simulate(line, horizon, warmup, seed=seed + r) for r in range(replications)]


def mean_and_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
