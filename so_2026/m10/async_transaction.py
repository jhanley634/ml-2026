import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

DATABASE_URL = "sqlite+aiosqlite:////tmp/atomic.db"


class DbMgr:
    _engine: AsyncEngine | None = None  # singleton

    @classmethod
    async def get_engine(cls) -> AsyncEngine:
        if cls._engine is None:
            cls._engine = create_async_engine(DATABASE_URL, future=True)
        return cls._engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession]:
    """Convenience async session manager."""
    async_session_factory = async_sessionmaker(bind=await DbMgr.get_engine())
    async with async_session_factory() as sess:
        try:
            yield sess
            await sess.commit()
        finally:
            await sess.close()


Base: declarative_base = declarative_base()


class Author(Base):
    __tablename__ = "author"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fname = Column(String, nullable=True)
    lname = Column(String, nullable=False)
    created_at = Column(Integer, default=lambda: int(time.time()))


class Book(Base):
    __tablename__ = "book"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    created_at = Column(Integer, default=lambda: int(time.time()))


async def init_db() -> None:
    engine = await DbMgr.get_engine()
    assert isinstance(engine, AsyncEngine)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def main() -> None:
    await init_db()


if __name__ == "__main__":
    asyncio.run(main())
