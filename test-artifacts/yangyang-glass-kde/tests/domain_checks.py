"""领域复算：原 KDE 结果、分台模型包与断网合并的一致性核对。"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))

from src.domain.measurements import demo_rows, normalize_rows  # noqa: E402
from src.domain.package_format import PackageError, merge_documents, seal  # noqa: E402
from src.domain.packaging import build_run_packages  # noqa: E402
from src.domain.project import analyze  # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    if condition:
        print(f'ok   {name}')
        return
    FAILURES.append(name)
    print(f'FAIL {name} {detail}')


def close(left, right, tolerance):
    return math.isclose(left, right, abs_tol=tolerance)


def main():
    rows = normalize_rows(demo_rows())
    result = analyze(rows, {})
    metrics = result['metrics']
    tolerance_rows = result['tables'][0]['rows']
    bandwidth = metrics['入选带宽（mm）']

    check('台账 144 条', len(rows) == 144, len(rows))
    check('入选带宽约 0.179', close(bandwidth, 0.179015, 1e-5), bandwidth)
    check('总超差约 30.52%', close(tolerance_rows[0]['outside_pct'], 30.5215, 1e-3))
    check('JY-C 上侧约 26.75%',
          close(next(row for row in tolerance_rows if row['bench'] == 'JY-C')['above_pct'], 26.7475, 1e-3))

    documents, table, complete = build_run_packages(rows, bandwidth, 24, 1e-4, 1, 18.9, 21.55)
    check('默认核上限三台均可发包', complete and len(documents) == 3, table)
    for row in table:
        check(f'{row["bench"]} 认证偏差不超过约定',
              row['certified_deviation'] is None or row['certified_deviation'] <= 1e-4)
        check(f'{row["bench"]} 核数不超过上限',
              row['kernel_count'] is None or row['kernel_count'] <= 24)
    check('包里不含逐件记录', all('diameter_mm' not in json.dumps(d, ensure_ascii=False)
                                  for d in documents.values()))

    serialized = [json.loads(json.dumps(documents[bench], ensure_ascii=False))
                  for bench in sorted(documents)]
    merged = merge_documents(serialized)
    check('合并全批超差与原模型一致',
          close(merged['full_batch']['outside_pct'], tolerance_rows[0]['outside_pct'], 5e-3),
          (merged['full_batch']['outside_pct'], tolerance_rows[0]['outside_pct']))
    for row, reference in zip(merged['benches'], tolerance_rows[1:]):
        check(f'{row["bench"]} 分台下侧一致',
              close(row['tolerance']['below_pct'], reference['below_pct'], 5e-3),
              (row['tolerance']['below_pct'], reference['below_pct']))
        check(f'{row["bench"]} 分台超差一致',
              close(100 - row['tolerance']['inside_pct'], reference['outside_pct'], 5e-3))

    narrow = merge_documents(serialized, lower=19.10, upper=21.40)
    check('窄窗全批超差 53.06%', close(narrow['full_batch']['outside_pct'], 53.06, 1e-2),
          narrow['full_batch']['outside_pct'])

    thresholds = merge_documents(serialized, thresholds=[19.0, 21.0])['full_batch']['thresholds']
    for row in thresholds:
        check(f'阈值 {row["threshold_mm"]:.2f} 两侧互补',
              close(row['below_pct'] + row['above_pct'], 100.0, 1e-9))

    for label, payload in (
        ('缺台', serialized[:2]),
        ('重复台', serialized + [serialized[0]]),
        ('混入另一运行', serialized[:2] + [seal({key: value for key, value in
                                              dict(serialized[2], run_id=9).items() if key != 'checksum'})]),
    ):
        try:
            merge_documents(payload)
        except PackageError:
            print(f'ok   拒绝{label}')
        else:
            check(f'拒绝{label}', False)

    capacity, _, refused = build_run_packages(rows, bandwidth, 4, 1e-4, 1, 18.9, 21.55)
    check('核数上限过小时拒绝发包', not refused and not capacity)

    print()
    if FAILURES:
        print(f'失败 {len(FAILURES)} 项：' + '、'.join(FAILURES))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
