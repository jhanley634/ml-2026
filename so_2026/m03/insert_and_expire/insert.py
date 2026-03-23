#! /usr/bin/env python

# from https://softwareengineering.stackexchange.com/questions/461005
# /how-should-we-purge-old-data-in-mysql-with-minimal-downtime

import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator


assert sys.version_info >= (3, 13)
assert sa.__version__ >= "2.0.45"

temp = Path("/tmp/k")
temp.mkdir(exist_ok=True)
db_file = temp / "bench.db"
_engine = create_engine(f"sqlite:///{db_file}")


class Base(DeclarativeBase):
    pass


@contextmanager
def get_session() -> Generator[Session]:
    with sessionmaker(bind=_engine)() as sess:
        try:
            yield sess
        finally:
            sess.commit()


# We only retain the last ~2 months of data in req_logs.
#
# req_logs
# id    client_txn_id  type   user_id   ad_id    created_at           updated_at           status
# 1     txn_001     forward   user_456  ad_123   2026-03-20 10:01:00  2026-03-20 10:01:00  processed
# 2     txn_002     refund    user_789  ad_123   2026-03-20 10:02:00  2026-03-20 10:02:00  processed


class ReqLog(Base):
    __tablename__ = "req_logs"

    id = Column(Integer, primary_key=True)
    client_txn_id = Column(String)
    type = Column(String)
    user_id = Column(String)
    ad_id = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    status = Column(String)


def create_empty_table() -> None:
    with get_session() as sess:

        Base.metadata.create_all(_engine)

        sess.query(ReqLog).delete()


def ins(num_recs: int = 100) -> None:
    """Uses sqlalchemy to INSERT a hundred rows into the req_log table."""

    with get_session() as sess:
        now = datetime.now(tz=UTC)

        for i in range(num_recs):
            row = ReqLog(
                client_txn_id=f"txn_{i:03d}",
                type="forward" if i % 2 == 0 else "refund",
                user_id=f"user_{i % 100}",
                ad_id=f"ad_{i % 50}",
                created_at=now,
                updated_at=now,
                status="processed",
            )
            sess.add(row)


if __name__ == "__main__":
    ...
