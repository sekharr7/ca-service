from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    orm,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from current_affairs.config import DB_URI

Base = declarative_base()


class CurrentAffair(Base):
    __tablename__ = "mtr_current_affairs_edk"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)
    theme = Column(String(100), nullable=True)
    theme_topic = Column(String(255), nullable=True)
    theme_subtopic = Column(String(255), nullable=True)
    ca_created_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    source = Column(String(500), nullable=True)
    view_count = Column(Integer, default=0)
    pinecone_synced = Column(Integer, default=0)


class Highlight(Base):
    __tablename__ = "mtr_ca_highlights_edk"

    id = Column(Integer, primary_key=True)
    current_affair_id = Column(Integer, nullable=False)
    highlight_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class Image(Base):
    __tablename__ = "mtr_ca_images_edk"

    id = Column(Integer, primary_key=True)
    current_affair_id = Column(Integer, nullable=False)
    image_url = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class Keyword(Base):
    __tablename__ = "mtr_ca_kword_edk"

    id = Column(Integer, primary_key=True)
    current_affair_id = Column(Integer, nullable=False)
    kword = Column(String(500), nullable=False)


class BulkJob(Base):
    __tablename__ = "mtr_ca_bulk_jobs"

    id = Column(String(36), primary_key=True)
    status = Column(String(32), default="pending")
    total = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    succeeded = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    details = Column(Text, default="[]")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


_engine = None
_SessionFactory = None


def get_db_session(custom_uri: str | None = None) -> orm.Session:
    """Create or retrieve a thread-safe SQLAlchemy database session."""
    global _engine, _SessionFactory
    uri = custom_uri or DB_URI
    if not uri:
        raise ValueError("DB_URI is not set in configuration or .env")

    if custom_uri or _engine is None:
        engine = create_engine(uri, pool_recycle=3600, pool_pre_ping=True)
        if not custom_uri:
            _engine = engine
            _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
            return _SessionFactory()
        else:
            CustomSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            return CustomSession()

    return _SessionFactory()
