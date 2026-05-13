"""
database/db.py — SQLAlchemy ORM setup with SQLite for the AI Checkout System.

Defines the four core tables (products, transactions, transaction_items,
inventory_log), the engine factory, and session management utilities.
All tables are created at startup via create_all_tables().
"""

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

Base = declarative_base()


# ──────────────────────────── ORM Models ────────────────────────────


class Product(Base):
    """Represents a grocery product in the inventory."""

    __tablename__ = "products"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(255), nullable=False, unique=True)
    category: str = Column(String(255), nullable=False)
    unit_price: float = Column(Float, nullable=False, default=0.0)
    stock_quantity: int = Column(Integer, nullable=False, default=0)
    barcode: str = Column(Text, nullable=True)
    image_url: str = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    # Relationships
    transaction_items = relationship("TransactionItem", back_populates="product")
    inventory_logs = relationship("InventoryLog", back_populates="product")

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name='{self.name}', price={self.unit_price}, stock={self.stock_quantity})>"


class Transaction(Base):
    """Represents a single checkout transaction."""

    __tablename__ = "transactions"

    id: str = Column(String(36), primary_key=True)  # UUID string
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_amount: float = Column(Float, nullable=False, default=0.0)
    tax_amount: float = Column(Float, nullable=False, default=0.0)
    subtotal: float = Column(Float, nullable=False, default=0.0)
    processed_image_path: str = Column(Text, nullable=True)
    processing_time_ms: float = Column(Float, nullable=True)

    # Relationships
    items = relationship("TransactionItem", back_populates="transaction", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Transaction(id='{self.id}', total={self.total_amount})>"


class TransactionItem(Base):
    """Line item within a checkout transaction."""

    __tablename__ = "transaction_items"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id: str = Column(String(36), ForeignKey("transactions.id"), nullable=False)
    product_id: int = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity: int = Column(Integer, nullable=False, default=1)
    unit_price_at_sale: float = Column(Float, nullable=False)
    subtotal: float = Column(Float, nullable=False)

    # Relationships
    transaction = relationship("Transaction", back_populates="items")
    product = relationship("Product", back_populates="transaction_items")

    def __repr__(self) -> str:
        return f"<TransactionItem(txn='{self.transaction_id}', product={self.product_id}, qty={self.quantity})>"


class InventoryLog(Base):
    """Audit trail for all inventory changes (sales and restocks)."""

    __tablename__ = "inventory_log"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.id"), nullable=False)
    change_type: str = Column(String(20), nullable=False)  # 'SALE' or 'RESTOCK'
    quantity_delta: int = Column(Integer, nullable=False)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    transaction_id: str = Column(String(36), ForeignKey("transactions.id"), nullable=True)

    # Relationships
    product = relationship("Product", back_populates="inventory_logs")

    def __repr__(self) -> str:
        return f"<InventoryLog(product={self.product_id}, type='{self.change_type}', delta={self.quantity_delta})>"


class AdminUser(Base):
    """Admin user account with SHA-256 hashed password and MFA support."""

    __tablename__ = "admin_users"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    username: str = Column(String(100), nullable=False, unique=True, index=True)
    email: str = Column(String(255), nullable=False, unique=True)
    password_hash: str = Column(String(64), nullable=False)  # SHA-256 hex digest
    salt: str = Column(String(32), nullable=False)  # Random salt for password hashing
    is_active: bool = Column(Boolean, default=True, nullable=False)
    mfa_enabled: bool = Column(Boolean, default=True, nullable=False)
    failed_attempts: int = Column(Integer, default=0, nullable=False)
    locked_until: datetime = Column(DateTime, nullable=True)  # Account lockout
    last_login: datetime = Column(DateTime, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<AdminUser(id={self.id}, username='{self.username}', active={self.is_active})>"


class OTPSession(Base):
    """Stores time-limited OTP codes for email-based multi-factor authentication."""

    __tablename__ = "otp_sessions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: int = Column(Integer, ForeignKey("admin_users.id"), nullable=False)
    otp_hash: str = Column(String(64), nullable=False)  # SHA-256 hash of the OTP code
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: datetime = Column(DateTime, nullable=False)
    is_used: bool = Column(Boolean, default=False, nullable=False)
    attempts: int = Column(Integer, default=0, nullable=False)  # Brute-force protection
    session_token: str = Column(String(64), nullable=False, unique=True)  # Pre-auth session ID

    # Relationships
    user = relationship("AdminUser")

    def __repr__(self) -> str:
        return f"<OTPSession(user_id={self.user_id}, used={self.is_used})>"


# ──────────────────────── Engine & Session ───────────────────────────


def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Enable WAL mode and foreign keys for SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def get_engine(db_path: str = "database/checkout.db"):
    """
    Create and return a SQLAlchemy engine for the given SQLite database path.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLAlchemy Engine instance.
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _set_sqlite_pragma)
    return engine


def get_session_factory(engine) -> sessionmaker:
    """
    Create and return a sessionmaker bound to the given engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        Configured sessionmaker class.
    """
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session(session_factory: sessionmaker) -> Generator[Session, None, None]:
    """
    Dependency generator that yields a database session and ensures cleanup.

    Args:
        session_factory: A configured sessionmaker instance.

    Yields:
        A SQLAlchemy Session.
    """
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def create_all_tables(engine) -> None:
    """
    Create all ORM-defined tables in the database if they don't already exist.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    Base.metadata.create_all(bind=engine)
