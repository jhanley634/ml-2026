#! /usr/bin/env python

from pathlib import Path
from time import time

import numpy as np
import polars as pl
from sqlalchemy import Boolean, Column, Engine, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_FILE = Path("/tmp/join.sqlite")


def get_engine() -> Engine:
    return create_engine(f"sqlite:///{DB_FILE}")


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()


class Base(DeclarativeBase): ...


class HugeDataFrame(Base):
    __tablename__ = "huge_df"

    id = Column(Integer, primary_key=True)
    a = Column(Integer, unique=True, index=True)  # Primary key or unique column
    huge_value = Column(Float)
    huge_category = Column(String(1))
    huge_score = Column(Float)
    huge_flag = Column(Boolean)
    huge_price = Column(Float)
    huge_rating = Column(Integer)
    huge_region = Column(String(5))
    huge_count = Column(Integer)
    huge_percentage = Column(Float)


class SmallDataFrame(Base):
    __tablename__ = "small_df"

    id = Column(Integer, primary_key=True)
    a = Column(Integer)  # Foreign key
    small_value = Column(Float)
    small_flag = Column(Boolean)
    small_category = Column(String(5))
    small_score = Column(Float)
    small_price = Column(Float)
    small_quantity = Column(Integer)
    small_status = Column(String(8))
    small_weight = Column(Float)
    small_temperature = Column(Float)
    small_pressure = Column(Float)
    small_density = Column(Float)
    small_ph = Column(Float)
    small_conductivity = Column(Float)
    small_viscosity = Column(Float)
    small_opacity = Column(Float)
    small_hardness = Column(Integer)
    small_color = Column(String(5))
    small_texture = Column(String(6))
    small_timestamp = Column(Integer)


# Print parquet file sizes
def format_bytes(bytes_size: float) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def get_column_names(model_class: Base) -> dict[str, None]:
    return {f"{c.key}": None for c in model_class.__table__.columns}


def get_huge_small(small_size: int = 500_000, huge_size: int = 2_000_000) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
]:
    # Create huge_df (already sorted on column 'A') with 10 columns
    rng = np.random.default_rng()
    huge_data = [
        {
            "a": i + 1,
            "huge_value": rng.standard_normal(),
            "huge_category": rng.choice(["X", "Y", "Z"]),
            "huge_score": rng.uniform(0, 100),
            "huge_flag": bool(rng.integers(2)),
            "huge_price": rng.exponential(50),
            "huge_rating": rng.integers(1, 6),  # inclusive of both ends
            "huge_region": rng.choice(["North", "South", "East", "West"]),
            "huge_count": rng.poisson(10),
            "huge_percentage": rng.beta(2, 5) * 100,
        }
        for i in range(huge_size)
    ]

    # Create small_df (unsorted on column 'A', with 80% overlap)
    overlap_size = int(small_size * 0.8)
    small_keys = list(
        rng.choice(huge_size, overlap_size, replace=False),
    ) + list(  # Keys that exist
        range(huge_size + 1, huge_size + small_size - overlap_size + 1),
    )  # Keys that don't
    rng.shuffle(small_keys)  # Make it unsorted

    small_data = [
        {
            "a": key,
            "small_value": rng.standard_normal(),
            "small_flag": bool(rng.integers(2)),
            "small_category": rng.choice(["Alpha", "Beta", "Gamma", "Delta"]),
            "small_score": rng.uniform(0, 1000),
            "small_price": rng.lognormal(4, 1),
            "small_quantity": rng.integers(1, 100),
            "small_status": rng.choice(["Active", "Inactive", "Pending"]),
            "small_weight": rng.gamma(2, 2),
            "small_temperature": rng.normal(20, 5),
            "small_pressure": rng.exponential(1.5),
            "small_density": rng.uniform(0.8, 1.2),
            "small_ph": rng.normal(7, 0.5),
            "small_conductivity": rng.exponential(10),
            "small_viscosity": rng.gamma(1.5, 2),
            "small_opacity": rng.beta(2, 3),
            "small_hardness": rng.integers(1, 11),  # inclusive of both ends
            "small_color": rng.choice(["Red", "Blue", "Green", "Yellow", "Orange"]),
            "small_texture": rng.choice(["Smooth", "Rough", "Grainy", "Silky"]),
            "small_timestamp": rng.integers(1600000000, 1700000000),
        }
        for key in small_keys
    ]

    return huge_data, small_data


def create_test_data() -> None:
    """Create sample data and save to SQLite database tables for realistic I/O testing"""

    start = time()
    huge_data, small_data = get_huge_small()

    engine = get_engine()
    Base.metadata.create_all(engine)
    session = get_session()

    print(1, int(time() - start))
    # huge_records = [HugeDataFrame(**record) for record in huge_data]
    huge_df = pl.DataFrame(huge_data)
    print(2, int(time() - start))
    # session.bulk_save_objects(huge_records)
    huge_df.write_database("huge_df", connection=engine, if_table_exists="replace")

    print(3, int(time() - start))
    # small_records = [SmallDataFrame(**record) for record in small_data]
    small_df = pl.DataFrame(small_data)
    print(4, int(time() - start))
    # session.bulk_save_objects(small_records)
    small_df.write_database("small_df", connection=engine, if_table_exists="replace")
    print(5, int(time() - start))

    session.commit()
    session.close()


def naive_approach(small_parquet_path: Path, huge_parquet_path: Path) -> pl.DataFrame:
    start_time = time()

    # Scan parquet files lazily - no data loaded into memory yet
    small_lf = pl.scan_parquet(small_parquet_path)
    huge_lf = pl.scan_parquet(huge_parquet_path)

    # Build query plan - Polars will optimize the entire pipeline
    result_lf = huge_lf.sort("A").join(small_lf, on="A", how="left")

    assert time() - start_time < 0.001

    # Execute optimized plan (includes I/O time)
    result_df = result_lf.collect()

    print(f"Join completed in {time() - start_time:.2f}s")
    return result_df


def main(*, want_cleanup: bool = False) -> None:
    print("Creating test data...")
    small_path = Path("temp_data/small_df.parquet")
    huge_path = Path("temp_data/huge_df.parquet")
    if not huge_path.exists():
        create_test_data()

    # Quick check of file sizes
    small_df_info = pl.scan_parquet(small_path).select(pl.len()).collect().item()
    huge_df_info = pl.scan_parquet(huge_path).select(pl.len()).collect().item()

    print(f"Small parquet: {small_df_info:,} rows (unsorted)")
    print(f"Huge parquet: {huge_df_info:,} rows (sorted)")

    print("\n--- Polars Sort and Join ---")
    result = naive_approach(small_path, huge_path)

    print(f"Result shape: {result.shape}")
    print(f"Successful joins: {result.filter(pl.col('huge_value').is_not_null()).height}")

    # Cleanup
    if want_cleanup:
        try:
            small_path.unlink()
            huge_path.unlink()
            huge_path.parent.rmdir()
            print("Cleaned up temporary files")
        except OSError:
            print(f"Note: Temporary files remain at {small_path.parent}")


if __name__ == "__main__":
    # Run: pip install polars numpy
    main()
