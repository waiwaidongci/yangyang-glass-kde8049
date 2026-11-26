"""运行的分台模型包：构建、持久化、下载与离线合并。"""

import logging

from sqlalchemy import select

from src.domain.package_format import MAX_BENCHES, PackageError, merge_documents
from src.domain.packaging import build_run_packages
from src.models.model_package import ModelPackage

logger = logging.getLogger(__name__)

PACKAGES_TABLE = 'packages'

_TABLE_LABELS = {
    'bench': '测径台', 'measurements': '原测件数', 'kernel_limit': '每包核上限',
    'kernel_count': '加权核数', 'certified_deviation': '认证单侧偏差',
    'deviation_limit': '约定偏差', 'status': '发包状态', 'note': '说明',
}


def packages_table(table_rows):
    return {
        'key': PACKAGES_TABLE, 'title': '分台口径模型包',
        'columns': list(_TABLE_LABELS), 'labels': _TABLE_LABELS, 'rows': table_rows,
    }


def build(session, rows, settings, run_id, bandwidth):
    """为一次运行生成并保存各台模型包，返回结果表（不改变运行状态）。"""
    documents, table_rows, complete = build_run_packages(
        rows, bandwidth, int(settings['package_kernel_limit']),
        float(settings['package_deviation_limit']), run_id,
        float(settings['tolerance_lower_mm']), float(settings['tolerance_upper_mm']),
    )
    for document in documents.values():
        session.add(ModelPackage(
            run_id=run_id, bench=document['bench'], measurements=document['measurements'],
            kernel_count=len(document['kernels']),
            certified_deviation=document['certified_deviation'],
            deviation_limit=document['deviation_limit'], checksum=document['checksum'],
            note='任一口径阈值处的单侧概率偏差不超过认证值', document=document,
        ))
    for row in table_rows:
        if row['bench'] in documents:
            continue
        session.add(ModelPackage(
            run_id=run_id, bench=row['bench'], measurements=row['measurements'],
            kernel_count=None, certified_deviation=None,
            deviation_limit=row['deviation_limit'], checksum=None,
            note=row['note'], document=None,
        ))
    session.flush()
    return packages_table(table_rows), complete


def listing(session, run_id):
    records = session.scalars(
        select(ModelPackage).where(ModelPackage.run_id == run_id).order_by(ModelPackage.bench)
    )
    return [
        {
            'bench': record.bench, 'measurements': record.measurements,
            'kernel_count': record.kernel_count, 'certified_deviation': record.certified_deviation,
            'deviation_limit': record.deviation_limit, 'checksum': record.checksum,
            'available': record.document is not None, 'note': record.note,
        }
        for record in records
    ]


def document(session, run_id, bench):
    record = session.get(ModelPackage, (run_id, bench))
    if record is None or record.document is None:
        raise LookupError('找不到可下载的口径模型包')
    return record.document


def merge(payload):
    """合并复测台带来的各台模型包；缺台、重复台或混入另一运行时拒绝出数。"""
    if not isinstance(payload, dict):
        raise PackageError('合并不出全批概率：请求体须为 JSON 对象')
    unknown = sorted(set(payload) - {'packages', 'lower_mm', 'upper_mm', 'thresholds_mm'})
    if unknown:
        raise PackageError(f'合并不出全批概率：未知字段 {unknown[0]}')
    documents = payload.get('packages')
    if not isinstance(documents, list) or not documents:
        raise PackageError('合并不出全批概率：未提供任何口径模型包')
    if len(documents) > MAX_BENCHES:
        raise PackageError(f'合并不出全批概率：一次最多合并 {MAX_BENCHES} 个模型包')
    thresholds = payload.get('thresholds_mm', [])
    if not isinstance(thresholds, list) or len(thresholds) > 64:
        raise PackageError('合并不出全批概率：阈值须为不超过 64 项的数值列表')
    return merge_documents(
        documents,
        lower=payload.get('lower_mm'), upper=payload.get('upper_mm'), thresholds=thresholds,
    )
