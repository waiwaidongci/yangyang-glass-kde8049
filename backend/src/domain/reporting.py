"""结果与审计的表结构，供报告、复核和 CSV 共用。"""

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np

from .kde import GaussianKDE
from .modes import GRADIENT_ULPS, POINTS_PER_BANDWIDTH, ROOT_TOLERANCE_MM


def _table(key, title, labels, rows):
    return {'key': key, 'title': title, 'columns': list(labels), 'labels': labels, 'rows': rows}


def tolerance_table(rows, model, lower, upper):
    groups = defaultdict(list)
    for row in rows:
        groups[row['bench']].append(row)

    result = []
    for bench, group in [(None, rows), *sorted(groups.items())]:
        scoped_model = model
        if bench is not None:
            samples = np.array([row['diameter_mm'] for row in group])
            scoped_model = GaussianKDE(samples, model.bandwidth)
        below = 100 * scoped_model.interval_mass(-math.inf, lower)
        inside = 100 * scoped_model.interval_mass(lower, upper)
        above = 100 * scoped_model.interval_mass(upper, math.inf)
        heldout = [row['diameter_mm'] for row in group if row['split'] == 'holdout']
        heldout_below = sum(value < lower for value in heldout)
        heldout_above = sum(value > upper for value in heldout)
        result.append({
            'scope': '全台账' if bench is None else '测径台', 'bench': bench,
            'n': len(group), 'below_pct': below, 'inside_pct': inside, 'above_pct': above,
            'outside_pct': below + above, 'holdout_n': len(heldout),
            'holdout_below': heldout_below, 'holdout_above': heldout_above,
            'holdout_outside_pct': 100 * (heldout_below + heldout_above) / len(heldout),
        })
    return _table('tolerance', '全台账与分台公差质量', {
        'scope': '范围', 'bench': '测径台', 'n': '重拟合测件数', 'below_pct': '估计低于下限（%）',
        'inside_pct': '估计公差内（%）', 'above_pct': '估计高于上限（%）', 'outside_pct': '估计超差（%）',
        'holdout_n': '留出测件数', 'holdout_below': '留出实测偏小数', 'holdout_above': '留出实测偏大数',
        'holdout_outside_pct': '留出实测超差（%）',
    }, result)


def modes_table(rows):
    return _table('modes', '密度峰与谷底证据', {
        'mode': '从小口径起的峰序', 'peak_mm': '峰位（mm）', 'peak_density': '峰高（1/mm）', 'flat_width_mm': '数值平顶宽度（mm）',
        'left_valley_mm': '左谷（mm）', 'right_valley_mm': '右谷（mm）',
        'valley_depth_pct': '较浅相邻谷相对峰的降幅（%）', 'basin_mass_pct': '谷间估计质量（%）',
        'observed_count': '谷间实测数', 'tolerance_position': '峰位与公差关系',
    }, rows)


def bandwidth_table(candidates, chosen_index):
    selected = candidates[chosen_index]
    return _table('bandwidth', '留出场次带宽评分', {
        'candidate': '候选序号', 'bandwidth_mm': '带宽（mm）',
        'mean_log_density': '留出平均对数密度', 'score_loss': '比入选值少', 'selected': '结果',
    }, [
        {'candidate': index + 1, 'bandwidth_mm': item.bandwidth, 'mean_log_density': item.mean_log_density,
         'score_loss': selected.mean_log_density - item.mean_log_density,
         'selected': '入选' if index == chosen_index else '候选'}
        for index, item in enumerate(candidates)
    ])


def density_table(model, points, lower, upper):
    left = min(lower, model.samples.min() - 4 * model.bandwidth)
    right = max(upper, model.samples.max() + 4 * model.bandwidth)
    marks = defaultdict(list)
    marks[lower].append('公差下限')
    marks[upper].append('公差上限')
    for point in points:
        marks[point.location].append('峰' if point.is_peak else '谷')
    coordinates = sorted(set(np.linspace(left, right, 241)) | marks.keys())
    return _table('density', '连续密度取样与峰谷标记', {
        'diameter_mm': '口径（mm）', 'density': '密度（1/mm）', 'mark': '定位',
    }, [
        {'diameter_mm': float(x), 'density': float(y), 'mark': '、'.join(marks.get(x, []))}
        for x, y in zip(coordinates, model.density(coordinates))
    ])


def sessions_table(rows):
    counts = Counter((row['session'], row['split'], row['bench']) for row in rows)
    return _table('sessions', '留出场次分配明细', {
        'session': '场次', 'split': '带宽评分用途', 'bench': '测径台', 'n': '测件数',
    }, [
        {'session': session, 'split': '拟合' if split == 'fit' else '留出', 'bench': bench, 'n': n}
        for (session, split, bench), n in sorted(counts.items())
    ])


def audit_table(rows, settings, candidates, chosen_index, grid_spacing):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    fit_sessions = {row['session'] for row in rows if row['split'] == 'fit'}
    heldout_sessions = {row['session'] for row in rows if row['split'] == 'holdout'}
    fit_count = sum(row['split'] == 'fit' for row in rows)
    values = {
        '台账 SHA-256': hashlib.sha256(payload).hexdigest(),
        '算法版本': 'gaussian-kde-holdout-v1',
        '拟合场次数': len(fit_sessions), '留出场次数': len(heldout_sessions),
        '两组共有场次数': len(fit_sessions & heldout_sessions),
        '拟合测件数': fit_count, '留出测件数': len(rows) - fit_count,
        '候选范围下限（mm）': settings['bandwidth_min_mm'], '候选范围上限（mm）': settings['bandwidth_max_mm'],
        '候选采样': f'{len(candidates)} 个等对数间距的绝对带宽，含两端',
        '带宽判定': '留出实测点的平均对数密度最高者入选，同分取较小带宽',
        '入选候选序号': chosen_index + 1,
        '入选带宽（mm）': candidates[chosen_index].bandwidth,
        '候选范围边界入选': '是' if chosen_index in (0, len(candidates) - 1) else '否',
        '最终重拟合测件数': len(rows), '最终重拟合方式': '固定带宽，以拟合与留出测件的并集估计密度',
        '公差下限（mm）': settings['tolerance_lower_mm'], '公差上限（mm）': settings['tolerance_upper_mm'],
        '峰谷扫描理论最大步长（mm）': candidates[chosen_index].bandwidth / POINTS_PER_BANDWIDTH,
        '峰谷扫描实际步长（mm）': grid_spacing, '过零求解绝对容差（mm）': ROOT_TOLERANCE_MM,
        '峰谷扫描舍入阈值': f'{GRADIENT_ULPS} 倍搜索边界浮点间隔除以带宽，应用于尺度化导数',
    }
    return _table('audit', '模型口径与复算审计', {'item': '核对项', 'value': '运行记录'}, [
        {'item': item, 'value': value} for item, value in values.items()
    ])
