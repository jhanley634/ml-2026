import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

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
    books = relationship("Book", back_populates="author")


class Book(Base):
    __tablename__ = "book"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    author_id = Column(Integer, ForeignKey("author.id"), nullable=False)
    created_at = Column(Integer, default=lambda: int(time.time()))
    author = relationship("Author", back_populates="books")


async def init_db() -> None:
    engine = await DbMgr.get_engine()
    assert isinstance(engine, AsyncEngine)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def add_player_of_games() -> None:
    async with get_session() as session:
        author = await session.execute(
            select(Author).where(Author.fname == "Iain M.", Author.lname == "Banks")
        )
        author = author.scalar_one_or_none()

        if author is None:
            author = Author(fname="Iain M.", lname="Banks")
            session.add(author)
            await session.flush()

        book = Book(title="The Player of Games", author=author)
        session.add(book)


async def main() -> None:
    await init_db()
    await add_player_of_games()


if __name__ == "__main__":
    asyncio.run(main())
