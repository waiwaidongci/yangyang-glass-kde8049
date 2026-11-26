"""一维等权高斯 KDE、留出似然和解析区间积分。"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, ndtr


CANDIDATE_COUNT = 17
_BLOCK_SIZE = 128
_LOG_SQRT_TWO_PI = 0.5 * math.log(2 * math.pi)


@dataclass(frozen=True)
class Candidate:
    bandwidth: float
    mean_log_density: float


def _blocks(values):
    for offset in range(0, len(values), _BLOCK_SIZE):
        yield values[offset:offset + _BLOCK_SIZE]


def select_bandwidth(fit, holdout, minimum, maximum):
    grid = np.geomspace(minimum, maximum, CANDIDATE_COUNT)
    bandwidths = np.unique(np.clip(grid, minimum, maximum))
    log_sums = np.zeros(len(bandwidths))
    for block in _blocks(holdout):
        squared_distances = np.square(block[:, None] - fit)
        for index, bandwidth in enumerate(bandwidths):
            weights = -squared_distances / (2 * bandwidth**2)
            log_sums[index] += logsumexp(weights, axis=1).sum()
    scores = log_sums / len(holdout) - np.log(bandwidths) - _LOG_SQRT_TWO_PI - math.log(len(fit))
    candidates = [Candidate(float(h), float(score)) for h, score in zip(bandwidths, scores)]
    return candidates, int(np.argmax(scores))


@dataclass(frozen=True)
class GaussianKDE:
    samples: np.ndarray
    bandwidth: float

    def density(self, locations):
        blocks = []
        for block in _blocks(np.atleast_1d(locations)):
            residuals = (self.samples - block[:, None]) / self.bandwidth
            blocks.append(np.exp(-0.5 * np.square(residuals)).mean(axis=1))
        return np.concatenate(blocks) / (self.bandwidth * math.exp(_LOG_SQRT_TWO_PI))

    def gradient_signals(self, locations):
        """计算 h 倍对数密度导数，避开稀疏区权重同时下溢。"""
        blocks = []
        for block in _blocks(np.atleast_1d(locations)):
            residuals = (self.samples - block[:, None]) / self.bandwidth
            log_weights = -0.5 * np.square(residuals)
            weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
            blocks.append((weights * residuals).sum(axis=1) / weights.sum(axis=1))
        return np.concatenate(blocks)

    def interval_mass(self, lower, upper):
        left = (lower - self.samples) / self.bandwidth
        right = (upper - self.samples) / self.bandwidth
        # 对正尾做生存函数相减，避免两个接近 1 的 CDF 相减。
        mass = np.where(left >= 0, ndtr(-left) - ndtr(-right), ndtr(right) - ndtr(left))
        return float(mass.mean())
