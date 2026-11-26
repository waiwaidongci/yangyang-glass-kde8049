"""把各台全批 KDE 压缩成有严格偏差保证的加权核模型包。

压缩沿用本次运行的原入选带宽：用不超过上限个加权高斯核近似该台的全批 KDE，
并给出整个实数轴上的偏差上界，保证任意口径阈值处的单侧概率与原模型之差不超过
约定值。容量内做不到就拒绝发包，绝不发出未达约定的模型包。包里只保存加权核，
不含逐件测量记录。
"""

import hashlib
import json
import math

import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr

from .package_format import (
    ALGORITHM, DEVIATION_LIMIT, DIAMETER_MAX, DIAMETER_MIN, FORMAT, FORMAT_VERSION,
    PackageError, seal, validate_document,
)

# |φ'| 的全局上界，用来把细网格采样推成整条实数轴的偏差界。
_PHI_PRIME_MAX = 1.0 / math.sqrt(2.0 * math.pi * math.e)
_MARGIN_BANDWIDTHS = 10.0
_MAX_CERTIFICATE_NODES = 600_000
_CANDIDATE_CEILING = 220
_REFINEMENT_ROUNDS = 4
_CHUNK = 512
_EPSILON = float(np.finfo(float).eps)


def _rounding_allowance(sample_count, kernel_count):
    """覆盖浮点求和、JSON 十进制往返与 BLAS 归约次序带来的舍入误差。"""
    return 8.0 * _EPSILON * (sample_count + kernel_count + 1)


class CapacityError(ValueError):
    """在每包加权核上限内无法满足约定偏差，拒绝发包。"""


def _blocked_cdf(nodes, centers, weights, bandwidth):
    values = np.empty(nodes.shape[0])
    for start in range(0, nodes.shape[0], _CHUNK):
        chunk = nodes[start:start + _CHUNK]
        values[start:start + chunk.shape[0]] = ndtr((chunk[:, None] - centers) / bandwidth) @ weights
    return values


def _original_cdf(samples, bandwidth, nodes):
    return _blocked_cdf(nodes, samples, np.full(samples.size, 1.0 / samples.size), bandwidth)


def _applied_cdf(centers, weights, bandwidth, nodes):
    return _blocked_cdf(nodes, centers, weights, bandwidth)


def _differences(samples, centers, weights, bandwidth, nodes):
    return (_applied_cdf(centers, weights, bandwidth, nodes)
            - _original_cdf(samples, bandwidth, nodes))


def _minimax(centers, bandwidth, nodes, original):
    """在节点上求最小最大绝对 CDF 偏差的加权核，返回权重与解得的节点偏差。"""
    matrix = ndtr((nodes[:, None] - centers) / bandwidth)
    count = matrix.shape[1]
    rows = matrix.shape[0]
    objective = np.zeros(count + 1)
    objective[-1] = 1.0
    constraints = np.vstack([
        np.hstack([matrix, -np.ones((rows, 1))]),
        np.hstack([-matrix, -np.ones((rows, 1))]),
    ])
    limits = np.concatenate([original, -original])
    equality = np.zeros((1, count + 1))
    equality[0, :count] = 1.0
    for method in ('highs', 'highs-ipm', 'highs-ds'):
        try:
            solution = linprog(objective, A_ub=constraints, b_ub=limits, A_eq=equality,
                               b_eq=[1.0], bounds=[(0.0, None)] * count + [(0.0, None)], method=method)
        except (ValueError, RuntimeError):
            continue
        if solution.success:
            weights = np.clip(np.asarray(solution.x[:count], dtype=float), 0.0, None)
            total = weights.sum()
            if total > 0:
                return weights / total, float(solution.x[-1])
    return None, None


def _tail(left_original, left_applied, right_original, right_applied):
    return max(float(left_original), float(left_applied),
               1.0 - float(right_original), 1.0 - float(right_applied), 0.0)


def _certificate(samples, centers, weights, bandwidth, limit):
    """在整个实数轴上给出 |近似 CDF − 原 CDF| 的严格上界。"""
    margin_width = _MARGIN_BANDWIDTHS * bandwidth
    left = float(min(samples.min(), centers.min())) - margin_width
    right = float(max(samples.max(), centers.max())) + margin_width
    # |D''| ≤ 2·φ'max/h²，故步长 s 的线性插值误差不超过 s²·φ'max/(4h²)；
    # 取 s = h·sqrt(limit/φ'max) 时这项恰为 limit/4。
    step = bandwidth * math.sqrt(limit / _PHI_PRIME_MAX)
    count = int(math.ceil((right - left) / step)) + 1
    if count > _MAX_CERTIFICATE_NODES:
        raise CapacityError(
            f'约定偏差 {limit:g} 所需核对网格超过 {_MAX_CERTIFICATE_NODES} 点，容量内无法认证'
        )
    nodes = np.linspace(left, right, count)
    original = _original_cdf(samples, bandwidth, nodes)
    applied = _applied_cdf(centers, weights, bandwidth, nodes)
    observed = float(np.abs(applied - original).max())
    interpolation = step * step * _PHI_PRIME_MAX / (4.0 * bandwidth * bandwidth)
    tail = _tail(original[0], applied[0], original[-1], applied[-1])
    rounding = _rounding_allowance(samples.size, centers.size)
    return observed + interpolation + tail + rounding


def _candidate_centers(samples, bandwidth, ceiling=_CANDIDATE_CEILING):
    """优先以实测口径为核中心，必要时补充等距网格点。"""
    low = max(DIAMETER_MIN, float(samples.min()) - bandwidth)
    high = min(DIAMETER_MAX, float(samples.max()) + bandwidth)
    measured = np.unique(samples)
    if measured.size > ceiling:
        picked = np.linspace(0, measured.size - 1, ceiling).round().astype(int)
        measured = measured[picked]
    if measured.size >= ceiling:
        return measured
    count = min(max(ceiling - measured.size, 16), ceiling)
    grid = np.linspace(low, high, count)
    merged = np.unique(np.concatenate([measured, grid]))
    if merged.size > ceiling:
        merged = np.linspace(low, high, ceiling)
    return merged


def _trim(centers, weights):
    order = np.argsort(centers)
    centers = np.clip(np.asarray(centers, dtype=float)[order], DIAMETER_MIN, DIAMETER_MAX)
    weights = np.clip(np.asarray(weights, dtype=float)[order], 0.0, None)
    weights = np.where(weights > 0, weights, 0.0)
    weights[weights < weights.max() * 1e-15] = 0.0
    total = weights.sum()
    if total <= 0:
        raise CapacityError('加权核方案的权重全为零，拒绝发包')
    keep = weights > 0
    return centers[keep], weights[keep] / total


def _probe(samples, bandwidth):
    low = float(samples.min()) - _MARGIN_BANDWIDTHS * bandwidth
    high = float(samples.max()) + _MARGIN_BANDWIDTHS * bandwidth
    return np.linspace(low, high, 2001)


def _probe_error(samples, centers, weights, bandwidth, probe):
    return _differences(samples, centers, weights, bandwidth, probe)


def _strategies(samples, bandwidth, nodes, original, candidates, limit, probe):
    """给出若干不超过上限个核的候选方案，按细网格实测偏差排序。"""
    solutions = []

    def add(centers, weights):
        if centers is None or weights is None or centers.size == 0 or centers.size > limit:
            return None
        centers, weights = _trim(centers, weights)
        error = float(np.abs(_probe_error(samples, centers, weights, bandwidth, probe)).max())
        solutions.append((error, centers, weights))
        return error

    weights, _ = _minimax(candidates, bandwidth, nodes, original)
    if weights is None:
        return solutions
    support = np.flatnonzero(weights > 1e-10)
    if support.size <= limit:
        add(candidates[support], weights[support])
        return solutions

    ranked = support[np.argsort(weights[support])[::-1]]
    kept, _ = _minimax(candidates[ranked[:limit]], bandwidth, nodes, original)
    add(candidates[ranked[:limit]], kept)

    # 分位中心：让近似 CDF 的等距分位点落在原模型上，作为独立的初始方案。
    quantiles = (np.arange(limit) + 0.5) / limit
    grid = np.linspace(float(nodes[0]), float(nodes[-1]), 2001)
    original_grid = _original_cdf(samples, bandwidth, grid)
    centers = np.array([float(np.interp(q, original_grid, grid)) for q in quantiles])
    centers = np.unique(centers)
    if 0 < centers.size <= limit:
        add(centers, _minimax(centers, bandwidth, nodes, original)[0])

    # 逆序消元只有在支持集仅略超上限时才划算。
    removable = support.size - limit
    if 0 < removable <= 16:
        order = list(ranked)
        current = None
        while len(order) > limit:
            trial, _ = _minimax(candidates[order[:-1]], bandwidth, nodes, original)
            if trial is None:
                break
            order = order[:-1]
            current = trial
        if current is not None and len(order) == limit:
            add(candidates[order], current)

    solutions.sort(key=lambda item: item[0])
    return solutions


def compress(samples, bandwidth, kernel_limit, deviation_limit):
    """返回该台的加权核近似与认证偏差；容量内做不到时抛出 CapacityError。"""
    values = np.sort(np.asarray(samples, dtype=float))
    if values.size == 0:
        raise CapacityError('该台没有测件，无法发包')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise CapacityError('原入选带宽无效，无法发包')
    if not math.isfinite(deviation_limit) or deviation_limit <= 0:
        raise CapacityError('约定偏差必须为正，无法发包')

    unique, counts = np.unique(values, return_counts=True)
    if unique.size <= kernel_limit:
        return {'centers': unique, 'weights': counts / values.size,
                'certified_deviation': 0.0, 'exact': True}

    candidates = _candidate_centers(values, bandwidth)
    probe = _probe(values, bandwidth)
    left = float(min(values.min(), candidates.min())) - _MARGIN_BANDWIDTHS * bandwidth
    right = float(max(values.max(), candidates.max())) + _MARGIN_BANDWIDTHS * bandwidth
    nodes = np.linspace(left, right, 1201)
    best_error = None

    for _ in range(_REFINEMENT_ROUNDS + 1):
        original = _original_cdf(values, bandwidth, nodes)
        solutions = _strategies(values, bandwidth, nodes, original, candidates, kernel_limit, probe)
        if not solutions:
            raise CapacityError('容量内未能求出可行的加权核方案，拒绝发包')
        error, centers, weights = solutions[0]
        if best_error is None or error < best_error:
            best_error = error
        bound = _certificate(values, centers, weights, bandwidth, deviation_limit)
        if bound <= deviation_limit:
            return {'centers': centers, 'weights': weights,
                    'certified_deviation': bound, 'exact': False}
        # 把细网格上偏差最大的位置补进节点和候选中心，再解一次。
        probe_differences = _probe_error(values, centers, weights, bandwidth, probe)
        additions = [float(probe[int(np.argmax(probe_differences))]),
                     float(probe[int(np.argmin(probe_differences))])]
        additions.extend(float(value) for value in centers[np.argsort(weights)[:2]])
        candidates = np.unique(np.concatenate([candidates, np.asarray(additions)]))
        if candidates.size > _CANDIDATE_CEILING + 40:
            candidates = np.linspace(min(values.min(), centers.min()) - bandwidth,
                                     max(values.max(), centers.max()) + bandwidth, _CANDIDATE_CEILING + 40)
        nodes = np.unique(np.concatenate([nodes, np.asarray(additions)]))
        if nodes.size > 2400:
            nodes = np.linspace(left, right, 2400)

    raise CapacityError(
        f'在每包最多 {kernel_limit} 个加权核内，单侧概率偏差只能压到约 {best_error + deviation_limit / 4:.3g}，'
        f'超过约定值 {deviation_limit:g}，拒绝发包'
    )


def dataset_digest(rows):
    """与运行审计一致的台账 SHA-256。"""
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _bench_document(bench, samples, measurements, roster, total, bandwidth, kernel_limit,
                    deviation_limit, run_id, dataset_checksum, lower, upper):
    model = compress(samples, bandwidth, kernel_limit, deviation_limit)
    document = seal({
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'run_id': int(run_id),
        'dataset_checksum': dataset_checksum,
        'algorithm': ALGORITHM,
        'bandwidth_mm': float(bandwidth),
        'kernel_limit': int(kernel_limit),
        'deviation_limit': float(deviation_limit),
        'tolerance_lower_mm': float(lower),
        'tolerance_upper_mm': float(upper),
        'total_measurements': int(total),
        'roster': list(roster),
        'bench': bench,
        'measurements': int(measurements),
        'kernels': [
            {'center_mm': float(center), 'weight': float(weight)}
            for center, weight in zip(model['centers'], model['weights'])
        ],
        'certified_deviation': float(model['certified_deviation']),
    })
    validate_document(document)
    return document


def build_run_packages(rows, bandwidth, kernel_limit, deviation_limit, run_id,
                       tolerance_lower, tolerance_upper):
    """为一次运行按台生成模型包。

    返回 ``(documents, table_rows, complete)``。达不到约定偏差的台不生成可下载
    文档（拒绝发包），其余台照常发；``complete`` 表示所有台都达约定。合并在
    缺台时同样不出全批概率。
    """
    if deviation_limit > DEVIATION_LIMIT:
        raise PackageError(f'约定偏差不得放宽到 {DEVIATION_LIMIT:g} 以上')
    benches = sorted({row['bench'] for row in rows})
    total = len(rows)
    digest = dataset_digest(rows)
    documents = {}
    table_rows = []
    complete = True
    for bench in benches:
        samples = np.array([row['diameter_mm'] for row in rows if row['bench'] == bench], dtype=float)
        try:
            document = _bench_document(
                bench, samples, samples.size, benches, total, bandwidth, kernel_limit,
                deviation_limit, run_id, digest, tolerance_lower, tolerance_upper,
            )
        except CapacityError as error:
            complete = False
            table_rows.append({
                'bench': bench, 'measurements': int(samples.size),
                'kernel_limit': int(kernel_limit), 'kernel_count': None,
                'certified_deviation': None, 'deviation_limit': float(deviation_limit),
                'status': '拒绝发包', 'note': str(error),
            })
            continue
        documents[bench] = document
        table_rows.append({
            'bench': bench, 'measurements': int(samples.size),
            'kernel_limit': int(kernel_limit), 'kernel_count': len(document['kernels']),
            'certified_deviation': document['certified_deviation'],
            'deviation_limit': float(deviation_limit),
            'status': '可下载', 'note': '任一口径阈值处的单侧概率偏差不超过认证值',
        })
    return documents, table_rows, complete
