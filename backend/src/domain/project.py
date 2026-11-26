"""共享台账、运行和复核入口使用的玻璃吹管领域接口。"""

import numpy as np

from .kde import GaussianKDE, select_bandwidth
from .measurements import DATASET_NAME, DIAMETER_MAX, DIAMETER_MIN, FIELDS
from .measurements import demo_rows, normalize_rows
from .modes import find_turning_points, mode_evidence
from .reporting import audit_table, bandwidth_table, density_table, modes_table, sessions_table, tolerance_table


SPEC = {
    'columns': list(FIELDS),
    'fields': FIELDS,
    'settings': [
        {'key': 'bandwidth_min_mm', 'label': '候选带宽下限（mm）', 'default': 0.05, 'min': 0.03, 'max': 4, 'step': 0.01},
        {'key': 'bandwidth_max_mm', 'label': '候选带宽上限（mm）', 'default': 1.5, 'min': 0.03, 'max': 4, 'step': 0.01},
        {'key': 'tolerance_lower_mm', 'label': '内口径公差下限（mm）', 'default': 18.9, 'min': DIAMETER_MIN,
         'max': DIAMETER_MAX, 'step': 0.01},
        {'key': 'tolerance_upper_mm', 'label': '内口径公差上限（mm）', 'default': 21.55, 'min': DIAMETER_MIN,
         'max': DIAMETER_MAX, 'step': 0.01},
    ],
    'modules': {'datasets': '测径台账', 'runs': '带宽与公差设置', 'results': '峰谷与公差报告', 'reviews': '复查交班'},
    'limitation': '仅分析这批内口径。留出评分用于选带宽，最终密度使用全批测件。峰位随带宽变化，公差概率供测径复查。',
    'dataset_name': DATASET_NAME,
    'recommendation': '核对留出场次和带宽评分，再按分台公差尾部、峰谷位置安排测径复查。',
}


def _settings(values):
    parameters = {item['key']: item for item in SPEC['settings']}
    if not isinstance(values, dict) or values.keys() - parameters.keys():
        raise ValueError('检测到未定义的测径运行参数')
    checked = {}
    for key, parameter in parameters.items():
        value = values.get(key, parameter['default'])
        if type(value) not in (int, float) or not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f'{parameter["label"]}须为 {parameter["min"]:g} 至 {parameter["max"]:g} 的有限数值')
        checked[key] = float(value)
    if checked['bandwidth_min_mm'] >= checked['bandwidth_max_mm']:
        raise ValueError('候选带宽下限须小于上限')
    if checked['tolerance_lower_mm'] >= checked['tolerance_upper_mm']:
        raise ValueError('内口径公差下限须小于上限')
    return checked


def analyze(rows, settings):
    checked = _settings(settings)
    rows = normalize_rows(rows)
    samples = np.array([row['diameter_mm'] for row in rows])
    fit_mask = np.array([row['split'] == 'fit' for row in rows])
    fit = samples[fit_mask]
    holdout = samples[~fit_mask]
    candidates, chosen_index = select_bandwidth(fit, holdout, checked['bandwidth_min_mm'], checked['bandwidth_max_mm'])
    chosen = candidates[chosen_index]
    model = GaussianKDE(samples, chosen.bandwidth)
    lower, upper = checked['tolerance_lower_mm'], checked['tolerance_upper_mm']
    points, grid_spacing = find_turning_points(model)
    modes = mode_evidence(model, points, lower, upper)
    tolerance = tolerance_table(rows, model, lower, upper)
    aggregate = tolerance['rows'][0]
    highest = max(tolerance['rows'][1:], key=lambda row: row['outside_pct'])
    warnings = ['留出场次参与了带宽选择。最终概率和峰谷使用全台账重拟合，留出实测比例仅供对照。']
    if chosen_index in (0, len(candidates) - 1):
        warnings.append('最优候选落在带宽范围边界，可调整候选范围再核对峰谷和尾部。')
    if len(modes) > 1 and min(row['valley_depth_pct'] for row in modes) < 10:
        warnings.append('有相邻峰的谷深不足峰高的 10%，请结合分台测径记录复查。')
    if any(point.is_peak and point.flat_width is not None for point in points):
        warnings.append('有数值平顶的峰，以平顶区间中点作为报告峰位。')
    bench_summary = f'分台中 {highest["bench"]} 的估计超差质量最高'
    if highest['below_pct'] != highest['above_pct']:
        side = '下侧' if highest['below_pct'] > highest['above_pct'] else '上侧'
        bench_summary += f'，主要在{side}'
    return {
        'summary': (
            f'以 {len(fit)} 条拟合测件建模，用 {len(holdout)} 条留出测件选出 {chosen.bandwidth:.3g} mm 带宽。'
            f'全台账重拟合检出 {len(modes)} 个密度峰，{lower:g}–{upper:g} mm 之外的估计质量为 '
            f'{aggregate["outside_pct"]:.2f}%。{bench_summary}。'
        ),
        'metrics': {
            '本批测件数': len(rows), '入选带宽（mm）': chosen.bandwidth,
            '入选留出平均对数密度': chosen.mean_log_density, '检出密度峰数': len(modes),
            '估计超差质量（%）': aggregate['outside_pct'],
        },
        'tables': [
            tolerance, modes_table(modes), bandwidth_table(candidates, chosen_index), density_table(model, points, lower, upper),
            audit_table(rows, checked, candidates, chosen_index, grid_spacing), sessions_table(rows),
        ],
        'warnings': warnings,
    }
