"""从解析导数的过零点提取峰、谷和测件复查区间。"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


POINTS_PER_BANDWIDTH = 16
ROOT_TOLERANCE_MM = 1e-10
GRADIENT_ULPS = 8


@dataclass(frozen=True)
class TurningPoint:
    location: float
    density: float
    is_peak: bool
    flat_width: float | None


def find_turning_points(model):
    bandwidth = model.bandwidth
    # 样本极值之外，所有核的导数同号。不必在长尾搜索极值。
    left = float(model.samples.min()) - bandwidth
    right = float(model.samples.max()) + bandwidth
    intervals = math.ceil((right - left) * POINTS_PER_BANDWIDTH / bandwidth)
    grid, spacing = np.linspace(left, right, num=intervals + 1, retstep=True)
    gradient = model.gradient_signals(grid)
    # 精确到机器舍入的平顶区间无法提供可靠的导数变号证据。
    noise_floor = GRADIENT_ULPS * np.spacing(max(abs(left), abs(right))) / bandwidth
    nonzero = np.flatnonzero(np.abs(gradient) > noise_floor)
    signs = np.sign(gradient[nonzero])
    flips = np.flatnonzero(signs[:-1] != signs[1:])
    locations = []
    flat_widths = []
    for index in flips:
        start, end = nonzero[index], nonzero[index + 1]
        if end - start >= 4:
            flat_left, flat_right = grid[start + 1], grid[end - 1]
            locations.append((flat_left + flat_right) / 2)
            flat_widths.append(float(flat_right - flat_left))
        else:
            locations.append(brentq(lambda x: model.gradient_signals(x)[0], grid[start], grid[end], xtol=ROOT_TOLERANCE_MM))
            flat_widths.append(None)
    points = [
        TurningPoint(float(location), float(density), bool(signs[index] > 0), width)
        for location, density, index, width in zip(locations, model.density(locations), flips, flat_widths)
    ]
    return points, float(spacing)


def _recheck_side(location, lower, upper):
    if location < lower:
        return '低于下限'
    if location > upper:
        return '高于上限'
    return '公差内'


def mode_evidence(model, points, lower, upper):
    peaks = [point for point in points if point.is_peak]
    valleys = [point for point in points if not point.is_peak]
    boundaries = [-math.inf, *(point.location for point in valleys), math.inf]
    local_counts = np.histogram(model.samples, bins=boundaries)[0]
    rows = []
    for index, peak in enumerate(peaks):
        left = valleys[index - 1] if index > 0 else None
        right = valleys[index] if index < len(valleys) else None
        adjacent = [point.density for point in (left, right) if point is not None]
        rows.append({
            'mode': index + 1,
            'peak_mm': peak.location,
            'peak_density': peak.density,
            'flat_width_mm': peak.flat_width,
            'left_valley_mm': left.location if left else None,
            'right_valley_mm': right.location if right else None,
            'valley_depth_pct': 100 * (1 - max(adjacent) / peak.density) if adjacent else None,
            'basin_mass_pct': 100 * model.interval_mass(boundaries[index], boundaries[index + 1]),
            'observed_count': int(local_counts[index]),
            'tolerance_position': _recheck_side(peak.location, lower, upper),
        })
    return rows
