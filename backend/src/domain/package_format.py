"""分台口径模型包的公开格式、校验与离线合并。

本模块只依赖标准库，供断网复测台直接调用：模型包里只有加权高斯核，不含逐件
测量记录。合并时按各台原测件数加权，得到全批公差概率。
"""

import hashlib
import hmac
import json
import math
from collections import Counter


FORMAT = 'yangyang-glass-kde.bench-model-package'
FORMAT_VERSION = 1
ALGORITHM = 'gaussian-kde-holdout-v1'
# 约定值：任一台在任意口径阈值下的单侧概率与原模型的最大绝对偏差。
DEVIATION_LIMIT = 1e-4
# 权重按十进制文本往返后允许的求和误差。
WEIGHT_SUM_TOLERANCE = 1e-9
DIAMETER_MIN = 5.0
DIAMETER_MAX = 60.0
# 单个模型包的核数上限硬顶，防止伪造包让离线合并卡死。
MAX_KERNEL_LIMIT = 256
MAX_BENCHES = 64
MAX_MEASUREMENTS = 5000
BANDWIDTH_MIN = 0.03
BANDWIDTH_MAX = 4.0
_SQRT_TWO = math.sqrt(2.0)

_REQUIRED = (
    'format', 'format_version', 'run_id', 'dataset_checksum', 'algorithm',
    'bandwidth_mm', 'kernel_limit', 'deviation_limit',
    'tolerance_lower_mm', 'tolerance_upper_mm',
    'total_measurements', 'roster', 'bench', 'measurements', 'kernels',
    'certified_deviation', 'checksum',
)
# 同一运行的模型包必须一致的字段。
_IDENTITY = (
    'dataset_checksum', 'algorithm', 'bandwidth_mm', 'kernel_limit', 'deviation_limit',
    'tolerance_lower_mm', 'tolerance_upper_mm', 'total_measurements',
)


class PackageError(ValueError):
    """模型包不完整、重复、混入另一运行或未达约定偏差。"""


def normal_cdf(value):
    return 0.5 * math.erfc(-value / _SQRT_TWO)


def canonical_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def checksum_of(payload):
    body = {key: value for key, value in payload.items() if key != 'checksum'}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal(payload):
    """补齐校验和；返回可直接下载或落盘的模型包。"""
    body = {key: value for key, value in payload.items() if key != 'checksum'}
    document = dict(body)
    document['checksum'] = checksum_of(body)
    return document


def package_cdf(document, threshold):
    """模型包在阈值处的下侧概率 P(口径 <= threshold)。"""
    bandwidth = document['bandwidth_mm']
    terms = (kernel['weight'] * normal_cdf((threshold - kernel['center_mm']) / bandwidth)
             for kernel in document['kernels'])
    return math.fsum(terms)


def _number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageError(f'模型包字段{label}不是数值，不输出全批概率')
    number = float(value)
    if not math.isfinite(number):
        raise PackageError(f'模型包字段{label}不是有限数值，不输出全批概率')
    if minimum is not None and number < minimum:
        raise PackageError(f'模型包字段{label}小于 {minimum:g}，不输出全批概率')
    if maximum is not None and number > maximum:
        raise PackageError(f'模型包字段{label}大于 {maximum:g}，不输出全批概率')
    return number


def _integer(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value != int(value):
        raise PackageError(f'模型包字段{label}不是整数，不输出全批概率')
    if minimum is not None and value < minimum:
        raise PackageError(f'模型包字段{label}小于 {minimum}，不输出全批概率')
    if maximum is not None and value > maximum:
        raise PackageError(f'模型包字段{label}大于 {maximum}，不输出全批概率')
    return int(value)


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise PackageError(f'模型包字段{label}不是有效文本，不输出全批概率')
    return value


def validate_document(document):
    """校验单个模型包的结构、权重与校验和，返回补全后的副本。"""
    if not isinstance(document, dict):
        raise PackageError('模型包不是 JSON 对象，不输出全批概率')
    missing = [key for key in _REQUIRED if key not in document]
    if missing:
        raise PackageError(f'模型包缺少字段 {missing[0]}，不输出全批概率')
    extra = sorted(set(document) - set(_REQUIRED))
    if extra:
        raise PackageError(f'模型包含未定义字段 {extra[0]}，不输出全批概率')
    if document['format'] != FORMAT:
        raise PackageError('模型包格式标识不符，不输出全批概率')
    if _integer(document['format_version'], 'format_version') != FORMAT_VERSION:
        raise PackageError(f'模型包格式版本须为 {FORMAT_VERSION}，不输出全批概率')
    try:
        stored = document['checksum']
        if not isinstance(stored, str) or not hmac.compare_digest(stored, checksum_of(document)):
            raise PackageError('模型包内容校验失败，可能已损坏或经过改动，不输出全批概率')
    except TypeError as error:
        raise PackageError('模型包内容无法校验，不输出全批概率') from error

    run_id = _integer(document['run_id'], 'run_id', minimum=1)
    dataset_checksum = _text(document['dataset_checksum'], 'dataset_checksum')
    if len(dataset_checksum) != 64 or any(ch not in '0123456789abcdef' for ch in dataset_checksum):
        raise PackageError('模型包的台账摘要格式有误，不输出全批概率')
    algorithm = _text(document['algorithm'], 'algorithm')
    if algorithm != ALGORITHM:
        raise PackageError(f'模型包算法版本不是 {ALGORITHM}，不输出全批概率')

    bandwidth = _number(document['bandwidth_mm'], 'bandwidth_mm',
                        minimum=BANDWIDTH_MIN, maximum=BANDWIDTH_MAX)
    kernel_limit = _integer(document['kernel_limit'], 'kernel_limit', minimum=1, maximum=MAX_KERNEL_LIMIT)
    deviation_limit = _number(document['deviation_limit'], 'deviation_limit', minimum=0.0)
    if deviation_limit > DEVIATION_LIMIT:
        raise PackageError(f'模型包声明的偏差上限超过约定值 {DEVIATION_LIMIT:g}，不输出全批概率')
    tolerance_lower = _number(document['tolerance_lower_mm'], 'tolerance_lower_mm')
    tolerance_upper = _number(document['tolerance_upper_mm'], 'tolerance_upper_mm')
    if tolerance_lower >= tolerance_upper:
        raise PackageError('模型包公差下限须小于上限，不输出全批概率')
    total_measurements = _integer(document['total_measurements'], 'total_measurements',
                                  minimum=1, maximum=MAX_MEASUREMENTS)
    measurements = _integer(document['measurements'], 'measurements',
                            minimum=1, maximum=MAX_MEASUREMENTS)
    if measurements > total_measurements:
        raise PackageError('模型包的台测件数大于全批测件数，不输出全批概率')

    roster = document['roster']
    if not isinstance(roster, list) or not roster:
        raise PackageError('模型包的测径台名录为空，不输出全批概率')
    if len(roster) > MAX_BENCHES:
        raise PackageError(f'模型包名录超过 {MAX_BENCHES} 个测径台，不输出全批概率')
    names = []
    for item in roster:
        names.append(_text(item, 'roster'))
    if len(set(names)) != len(names):
        raise PackageError('模型包名录中有重复测径台，不输出全批概率')
    bench = _text(document['bench'], 'bench')
    if bench not in names:
        raise PackageError('模型包的测径台不在名录中，不输出全批概率')

    kernels = document['kernels']
    if not isinstance(kernels, list) or not kernels:
        raise PackageError(f'测径台 {bench} 的模型包没有加权核，不输出全批概率')
    if len(kernels) > kernel_limit:
        raise PackageError(f'测径台 {bench} 的加权核数超过每包上限 {kernel_limit}，不输出全批概率')
    total_weight = 0.0
    checked_kernels = []
    for index, kernel in enumerate(kernels, start=1):
        if not isinstance(kernel, dict) or set(kernel) != {'center_mm', 'weight'}:
            raise PackageError(f'测径台 {bench} 第 {index} 个加权核字段有误，不输出全批概率')
        center = _number(kernel['center_mm'], 'center_mm',
                         minimum=DIAMETER_MIN - 1e-6, maximum=DIAMETER_MAX + 1e-6)
        weight = _number(kernel['weight'], 'weight', minimum=0.0)
        total_weight += weight
        checked_kernels.append({'center_mm': center, 'weight': weight})
    if abs(total_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise PackageError(f'测径台 {bench} 的模型包权重之和不为 1，不输出全批概率')
    certified = _number(document['certified_deviation'], 'certified_deviation', minimum=0.0)
    if certified > deviation_limit:
        raise PackageError(f'测径台 {bench} 的模型包超出声明的偏差上限，不输出全批概率')

    return {
        'format': FORMAT, 'format_version': FORMAT_VERSION, 'run_id': run_id,
        'dataset_checksum': dataset_checksum, 'algorithm': algorithm,
        'bandwidth_mm': bandwidth, 'kernel_limit': kernel_limit, 'deviation_limit': deviation_limit,
        'tolerance_lower_mm': tolerance_lower, 'tolerance_upper_mm': tolerance_upper,
        'total_measurements': total_measurements, 'roster': names, 'bench': bench,
        'measurements': measurements, 'kernels': checked_kernels,
        'certified_deviation': certified, 'checksum': document['checksum'],
    }


def _joined(values):
    return '、'.join(str(value) for value in values)


def merge_documents(documents, lower=None, upper=None, thresholds=()):
    """合并同一运行的各台模型包，返回全批公差概率。

    缺台、重复台、混入另一运行或台级偏差超限时抛出 PackageError，绝不返回
    全批概率。``thresholds`` 为额外的任意口径阈值，逐台和全批都给出单侧概率。
    """
    if not documents:
        raise PackageError('没有收到任何口径模型包，不输出全批概率')
    checked = [validate_document(document) for document in documents]

    benches = [document['bench'] for document in checked]
    repeated = sorted(name for name, count in Counter(benches).items() if count > 1)
    if repeated:
        raise PackageError(f'测径台 {_joined(repeated)} 的模型包重复，不输出全批概率')

    runs = sorted({document['run_id'] for document in checked})
    if len(runs) > 1:
        raise PackageError(f'混入了另一运行的模型包（运行 {_joined(runs)}），不输出全批概率')
    reference = checked[0]
    if any(document['roster'] != reference['roster'] for document in checked):
        raise PackageError('各模型包的测径台名录不一致，不输出全批概率')
    roster = sorted(reference['roster'])
    missing = [name for name in roster if name not in benches]
    extra = [name for name in benches if name not in roster]
    if missing:
        raise PackageError(f'缺少测径台 {_joined(missing)} 的模型包，不输出全批概率')
    if extra:
        raise PackageError(f'混入了名录之外的测径台 {_joined(extra)}，不输出全批概率')

    for document in checked[1:]:
        for key in _IDENTITY:
            if document[key] != reference[key]:
                raise PackageError('各模型包的运行参数不一致，可能混入了另一运行，不输出全批概率')
    total = reference['total_measurements']
    counted = sum(document['measurements'] for document in checked)
    if counted != total:
        raise PackageError(f'各台测件数之和 {counted} 与原测件数 {total} 不符，不输出全批概率')

    tolerance_lower = reference['tolerance_lower_mm'] if lower is None else _number(lower, 'lower')
    tolerance_upper = reference['tolerance_upper_mm'] if upper is None else _number(upper, 'upper')
    if tolerance_lower >= tolerance_upper:
        raise PackageError('公差下限须小于上限，不输出全批概率')
    extra_thresholds = [_number(value, 'threshold') for value in thresholds]

    ordered = sorted(checked, key=lambda document: document['bench'])
    per_bench = []
    for document in ordered:
        weight = document['measurements'] / total
        cdf_lower = package_cdf(document, tolerance_lower)
        cdf_upper = package_cdf(document, tolerance_upper)
        threshold_rows = []
        for threshold in extra_thresholds:
            cdf = package_cdf(document, threshold)
            threshold_rows.append({'threshold_mm': threshold, 'below_pct': 100 * cdf,
                                   'above_pct': 100 * (1 - cdf)})
        per_bench.append({
            'bench': document['bench'], 'measurements': document['measurements'],
            'share_pct': 100 * weight, 'kernel_count': len(document['kernels']),
            'certified_deviation': document['certified_deviation'],
            'tolerance': {'below_pct': 100 * cdf_lower,
                          'inside_pct': 100 * (cdf_upper - cdf_lower),
                          'above_pct': 100 * (1 - cdf_upper)},
            'thresholds': threshold_rows,
        })

    def weighted(values):
        return math.fsum((document['measurements'] / total) * value
                         for document, value in zip(ordered, values))

    cdfs_lower = [package_cdf(document, tolerance_lower) for document in ordered]
    cdfs_upper = [package_cdf(document, tolerance_upper) for document in ordered]
    full_lower = weighted(cdfs_lower)
    full_upper = weighted(cdfs_upper)
    full_thresholds = []
    for threshold in extra_thresholds:
        cdf = weighted([package_cdf(document, threshold) for document in ordered])
        full_thresholds.append({'threshold_mm': threshold, 'below_pct': 100 * cdf,
                                'above_pct': 100 * (1 - cdf)})
    guaranteed = weighted([document['certified_deviation'] for document in ordered])

    return {
        'ok': True,
        'run_id': reference['run_id'],
        'dataset_checksum': reference['dataset_checksum'],
        'algorithm': reference['algorithm'],
        'bandwidth_mm': reference['bandwidth_mm'],
        'deviation_limit': reference['deviation_limit'],
        'guaranteed_deviation': guaranteed,
        'total_measurements': total,
        'roster': roster,
        'tolerance_lower_mm': tolerance_lower,
        'tolerance_upper_mm': tolerance_upper,
        'benches': per_bench,
        'full_batch': {
            'measurements': total,
            'below_pct': 100 * full_lower,
            'inside_pct': 100 * (full_upper - full_lower),
            'above_pct': 100 * (1 - full_upper),
            'outside_pct': 100 * (full_lower + 1 - full_upper),
            'thresholds': full_thresholds,
        },
    }
