# Source - https://stackoverflow.com/q/80012706
# Posted by madjetmax
# Retrieved 2026-10-26, License - CC BY-SA 4.0

import asyncio
import uuid
from contextlib import asynccontextmanager
from time import time
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import JSON, UUID, Column, Integer, String, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import literal

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

assert sa.__version__ == "2.1.6"

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


class GroupChatMember(Base):
    __tablename__ = "group_chat_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    role = Column(JSON, nullable=True)
    group_key = Column(String, nullable=True)
    created_at = Column(Integer, default=lambda: int(time()))
    updated_at = Column(
        Integer,
        default=lambda: int(time()),
        onupdate=lambda: int(time()),
    )


async def add_group_member_session_group_key(
    session: AsyncSession,
    group_id: int,
    user_id: int,
    user_session_id: str,
    session_group_key: str,
) -> None:
    query = (
        update(GroupChatMember)
        .where(
            GroupChatMember.group_id == group_id,
            GroupChatMember.user_id == user_id,
        )
        .values(
            group_key=GroupChatMember.group_key
            + literal({user_session_id: session_group_key}, JSON),
        )
    )
    await session.execute(query)


async def init_db() -> None:
    async_engine = await DbMgr.get_engine()
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def main() -> None:
    await init_db()
    async with get_session() as sess:
        stmt = update(GroupChatMember).where(GroupChatMember.id == 1).values(role={"key": "value"})
        await sess.execute(stmt)
        await sess.commit()

        await add_group_member_session_group_key(
            session=sess,
            group_id=123,
            user_id=456,
            user_session_id=str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
            session_group_key="some_key",
        )


if __name__ == "__main__":
    asyncio.run(main())
