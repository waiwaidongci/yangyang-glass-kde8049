"""吹管测径台账与按场次留出的输入约束。"""

import math
import re
from collections import Counter
from random import Random


FIELDS = {
    'id': '吹管测件编号',
    'bench': '测径台',
    'session': '采样场次',
    'split': '拟合或留出（fit / holdout）',
    'diameter_mm': '吹管内口径（mm）',
}
DEMO_SEED = 261138
DATASET_NAME = '涧焰吹管口径_2026-11-25.csv'
DIAMETER_MIN = 5.0
DIAMETER_MAX = 60.0
MAX_ROWS = 5000
MIN_FIT = 16
MIN_HOLDOUT = 8
IDENTIFIER = re.compile(r'[^\W_][\w-]*')


def demo_rows():
    rng = Random(DEMO_SEED)
    benches = (('JY-A', 4, -0.05), ('JY-B', 3, 0.02), ('JY-C', 2, 0.08))
    sessions = ('S0720', 'S0740', 'S0800', 'S0820', 'S0840', 'S0900', 'S0920', 'S0940')
    rows = []
    for session_index, session in enumerate(sessions):
        session_shift = rng.uniform(-0.05, 0.05)
        for bench, narrow_count, bench_shift in benches:
            sizes = [0] * narrow_count + [1] * (6 - narrow_count)
            rng.shuffle(sizes)
            for size in sizes:
                center, spread = ((18.98, 0.20), (21.32, 0.25))[size]
                rows.append({
                    'id': f'JW-20261125-{len(rows) + 1:04d}',
                    'bench': bench,
                    'session': session,
                    'split': 'fit' if session_index < 6 else 'holdout',
                    'diameter_mm': round(rng.gauss(center + bench_shift + session_shift, spread), 2),
                })
    return rows


def _identifier(value, field, line, max_length):
    if isinstance(value, str):
        value = value.strip()
        if len(value) <= max_length and IDENTIFIER.fullmatch(value):
            return value
    raise ValueError(f'第 {line} 行{FIELDS[field]}应为 1 至 {max_length} 位文字、数字、下划线或连字符，且以文字或数字开头')


def normalize_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('台账没有测径记录')
    if len(rows) > MAX_ROWS:
        raise ValueError(f'单个测径台账最多 {MAX_ROWS} 行')

    seen = set()
    session_splits = {}
    bench_splits = Counter()
    split_counts = Counter()
    normalized = []
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, dict) or row.keys() != FIELDS.keys():
            raise ValueError(f'第 {line} 行字段必须为 ' + '、'.join(FIELDS))
        tube_id = _identifier(row['id'], 'id', line, 64)
        if tube_id in seen:
            raise ValueError(f'第 {line} 行测件编号 {tube_id} 重复，每根吹管只应提供一次测量')
        seen.add(tube_id)
        bench = _identifier(row['bench'], 'bench', line, 24)
        session = _identifier(row['session'], 'session', line, 32)
        raw_split = row['split']
        split = raw_split.strip() if isinstance(raw_split, str) else ''
        if split not in ('fit', 'holdout'):
            raise ValueError(f'第 {line} 行采样用途须填 fit 或 holdout')
        if session_splits.setdefault(session, split) != split:
            raise ValueError(f'采样场次 {session} 同时出现在拟合和留出中，请整场分组')
        raw_value = row['diameter_mm']
        try:
            if type(raw_value) not in (str, int, float):
                raise ValueError
            diameter = float(raw_value)
            if not math.isfinite(diameter) or not DIAMETER_MIN <= diameter <= DIAMETER_MAX:
                raise ValueError
        except (ValueError, OverflowError) as error:
            raise ValueError(f'第 {line} 行内口径须为 {DIAMETER_MIN:g} 至 {DIAMETER_MAX:g} mm 的有限数值') from error
        normalized.append({'id': tube_id, 'bench': bench, 'session': session, 'split': split, 'diameter_mm': diameter})
        bench_splits[bench, split] += 1
        split_counts[split] += 1

    if split_counts['fit'] < MIN_FIT or split_counts['holdout'] < MIN_HOLDOUT:
        raise ValueError(f'至少需要 {MIN_FIT} 条拟合测量和 {MIN_HOLDOUT} 条留出测量')
    for bench in sorted({row['bench'] for row in normalized}):
        if bench_splits[bench, 'fit'] < 4 or bench_splits[bench, 'holdout'] < 2:
            raise ValueError(f'测径台 {bench} 至少需要 4 条拟合测量和 2 条留出测量')
    return sorted(normalized, key=lambda row: row['id'])
