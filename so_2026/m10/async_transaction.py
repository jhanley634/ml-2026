import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

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
    async_session_factory = async_sessionmaker(bind=await DbMgr.get_engine())
    async with async_session_factory() as sess:
        try:
            yield sess
            await sess.commit()
        finally:
            await sess.close()


class Base(DeclarativeBase):
    pass


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


async def add_book(author: str, title: str) -> None:
    async with get_session() as session:
        fname, lname = author.rsplit(maxsplit=1)
        q = select(Author).where(
            Author.fname == fname,
            Author.lname == lname,
        )
        author_row = (await session.execute(q)).scalar_one_or_none()

        if author_row is None:
            author_row = Author(fname=fname, lname=lname)
            session.add(author_row)
            await session.flush()

        book = Book(title=title, author=author_row)
        session.add(book)


async def main() -> None:
    await init_db()
    async with get_session() as sess:
        await sess.execute(delete(Book))

    await add_book("Iain M. Banks", "Consider Phlebas")
    await add_book("Iain M. Banks", "The Player of Games")
    await add_book("Isaac Asimov", "Foundation")


if __name__ == "__main__":
    asyncio.run(main())
