# Source - https://stackoverflow.com/q/80012706
# Posted by madjetmax
# Retrieved 2026-10-26, License - CC BY-SA 4.0

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import JSON, UUID, Column, Engine, Integer, String, __version__, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import literal

if TYPE_CHECKING:
    from sqlalchemy.orm.decl_api import DeclarativeMeta

assert __version__ == "2.1.6"


DATABASE_URL = "sqlite+aiosqlite:///tmp/atomic.db?uri=true&echo=true"


class DbMgr:
    _engine: Engine | None = None  # singleton

    @classmethod
    def get_engine(cls) -> Engine:
        if cls._engine is None:
            cls._engine = create_async_engine(DATABASE_URL, future=True)
        return cls._engine


@asynccontextmanager
async def get_session() -> AsyncSession:
    """Convenience async session manager."""
    async_session_factory = async_sessionmaker(bind=DbMgr.get_engine(), expire_on_commit=False)
    async with async_session_factory() as sess:
        try:
            yield sess
            await sess.commit()
        finally:
            await sess.close()


Base: DeclarativeMeta = declarative_base()


class GroupChatMember(Base):
    __tablename__ = "group_chat_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    role = Column(JSON, nullable=True)
    group_key = Column(String, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
    )


async def add_group_member_session_group_key(
    session: AsyncSession,
    group_id: int,
    user_id: int,
    user_session_id: uuid.UUID,
    session_group_key: tuple,
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
    await session.commit()


async def async_db_session() -> AsyncEngine | Engine:
    engine = await async_sessionmaker(bind=DbMgr.get_engine())
    print(type(engine))
    return engine


async def init_db() -> None:
    async_engine = DbMgr.get_engine()
    await async_engine.run_sync(Base.metadata.create_all)


async def main() -> None:

    # engine = create_async_engine("postgresql+asyncpg://user:password@localhost/dbname")

    await init_db()

    async with AsyncSession(bind=DbMgr.get_engine()) as sess:

        stmt = (
            update(GroupChatMember)
            .where(GroupChatMember.id == 1)
            .values(json_data={"key": "value"})
        )
        await sess.execute(stmt)
        await sess.commit()

        await add_group_member_session_group_key(
            session=sess,
            group_id=123,
            user_id=456,
            user_session_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            session_group_key="some_key",
        )


if __name__ == "__main__":
    asyncio.run(main())
