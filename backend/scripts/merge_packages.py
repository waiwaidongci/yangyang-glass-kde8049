"""断网复测台合并各台口径模型包，按原测件数算全批公差概率。

只依赖标准库，可直接把各台下载的 JSON 文件放在一起运行：

    python3 scripts/merge_packages.py JY-A.json JY-B.json JY-C.json
    python3 scripts/merge_packages.py *.json --lower 19.10 --upper 21.40 \
        --threshold 19.50 --threshold 20.80

缺台、重复台、混入另一运行或任一台未达约定偏差时，脚本以非零状态退出，
不输出全批概率。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.package_format import PackageError, merge_documents  # noqa: E402


def _load(paths):
    documents = []
    for path in paths:
        try:
            documents.append(json.loads(Path(path).read_text(encoding='utf-8')))
        except OSError as error:
            raise PackageError(f'读不到模型包 {path}：{error.strerror}，不输出全批概率') from error
        except json.JSONDecodeError as error:
            raise PackageError(f'模型包 {path} 不是有效 JSON，不输出全批概率') from error
    return documents


def _format(report):
    lines = [
        f"运行 {report['run_id']}，台账摘要 {report['dataset_checksum'][:16]}…，"
        f"入选带宽 {report['bandwidth_mm']:.6g} mm",
        f"原测件数 {report['total_measurements']}，合并 {len(report['benches'])} 台，"
        f"保证偏差不超过 {report['guaranteed_deviation']:.3g}",
        '',
        f"{'范围':<6}{'测径台':<10}{'原测件数':>8}{'占比%':>8}{'低于下限%':>11}{'公差内%':>10}"
        f"{'高于上限%':>11}{'超差%':>9}",
    ]
    for item in report['benches']:
        tolerance = item['tolerance']
        lines.append(
            f"{'分台':<6}{item['bench']:<10}{item['measurements']:>8}{item['share_pct']:>8.2f}"
            f"{tolerance['below_pct']:>11.4f}{tolerance['inside_pct']:>10.4f}"
            f"{tolerance['above_pct']:>11.4f}{100 - tolerance['inside_pct']:>9.4f}"
        )
    full = report['full_batch']
    lines.append(
        f"{'全批':<6}{'—':<10}{full['measurements']:>8}{100.0:>8.2f}"
        f"{full['below_pct']:>11.4f}{full['inside_pct']:>10.4f}"
        f"{full['above_pct']:>11.4f}{full['outside_pct']:>9.4f}"
    )
    if full['thresholds']:
        lines += ['', f"{'阈值mm':>10}{'全批低于该口径%':>18}{'全批高于该口径%':>18}"]
        for row in full['thresholds']:
            lines.append(f"{row['threshold_mm']:>10.4f}{row['below_pct']:>18.4f}{row['above_pct']:>18.4f}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='断网合并各台口径模型包，算全批公差概率')
    parser.add_argument('packages', nargs='+', help='各台下载的模型包 JSON 路径')
    parser.add_argument('--lower', type=float, default=None, help='公差下限（mm），默认取包内值')
    parser.add_argument('--upper', type=float, default=None, help='公差上限（mm），默认取包内值')
    parser.add_argument('--threshold', action='append', type=float, default=[],
                        help='附加口径阈值（mm），可重复')
    parser.add_argument('--json', action='store_true', help='输出完整 JSON 而不是表格')
    arguments = parser.parse_args(argv)

    try:
        report = merge_documents(_load(arguments.packages), lower=arguments.lower,
                                 upper=arguments.upper, thresholds=arguments.threshold)
    except PackageError as error:
        print(str(error), file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
