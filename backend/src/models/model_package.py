from sqlalchemy import JSON, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class ModelPackage(Base):
    """一次运行的某台口径模型包；文档里只有加权核，不含逐件测量记录。"""

    __tablename__ = 'model_packages'

    run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id'), primary_key=True)
    bench: Mapped[str] = mapped_column(primary_key=True)
    measurements: Mapped[int] = mapped_column(Integer)
    kernel_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    certified_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    deviation_limit: Mapped[float] = mapped_column(Float)
    checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str] = mapped_column(Text)
    document: Mapped[dict | None] = mapped_column(JSON, nullable=True)
