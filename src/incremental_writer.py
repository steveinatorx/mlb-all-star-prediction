"""Incremental file writing for resilient data ingestion."""

import json
from pathlib import Path
from typing import Any, Optional

import polars as pl
from loguru import logger


class IncrementalWriter:
    """Writes data incrementally to JSONL files, then converts to Parquet."""

    def __init__(self, output_path: Path, batch_size: int = 100):
        """
        Initialize incremental writer.

        Args:
            output_path: Final parquet output path
            batch_size: Number of records to accumulate before writing to JSONL
        """
        self.output_path = output_path
        self.batch_size = batch_size
        self.jsonl_path = output_path.with_suffix(".jsonl")
        self.buffer: list[dict[str, Any]] = []
        self.total_written = 0

    def write_row(self, row: dict[str, Any]) -> None:
        """
        Write a single row incrementally.

        Args:
            row: Dictionary representing a single record
        """
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def write_batch(self, rows: list[dict[str, Any]]) -> None:
        """
        Write multiple rows at once.

        Args:
            rows: List of dictionaries representing records
        """
        self.buffer.extend(rows)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffer to JSONL file."""
        if not self.buffer:
            return

        # Append to JSONL file (atomic append)
        mode = "a" if self.jsonl_path.exists() else "w"
        with open(self.jsonl_path, mode) as f:
            for row in self.buffer:
                f.write(json.dumps(row) + "\n")

        self.total_written += len(self.buffer)
        logger.debug(f"Wrote {len(self.buffer)} records to {self.jsonl_path} (total: {self.total_written})")
        self.buffer.clear()

    def finalize(self) -> Path:
        """
        Flush remaining buffer and convert JSONL to Parquet.

        Returns:
            Path to final parquet file
        """
        # Flush any remaining buffer
        self.flush()

        if not self.jsonl_path.exists():
            logger.warning(f"No data written to {self.jsonl_path}")
            return self.output_path

        # Read JSONL and convert to Parquet
        # Count actual records in file
        record_count = 0
        with open(self.jsonl_path, "r") as f:
            record_count = sum(1 for line in f if line.strip())
        
        logger.info(f"Converting {self.jsonl_path} to Parquet ({record_count} records)...")
        
        try:
            # Read JSONL line by line to handle large files
            rows = []
            with open(self.jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line: {e}")

            if not rows:
                logger.warning("No data to convert")
                return self.output_path

            # Convert to DataFrame with better schema handling
            # Use scan_ndjson with higher infer_schema_length to avoid type mismatches
            logger.info("Reading JSONL with schema inference...")
            try:
                # Try reading with scan_ndjson (lazy, more memory efficient)
                df = pl.read_ndjson(
                    self.jsonl_path,
                    infer_schema_length=50000,  # Sample more rows for schema inference
                    ignore_errors=False,
                )
                logger.info(f"Successfully read {len(df)} records")
            except Exception as scan_error:
                logger.warning(f"Direct read failed: {scan_error}, trying alternative method...")
                # Fallback: read in chunks and handle schema manually
                df = pl.DataFrame(rows)
                # Coerce common problematic columns to proper types
                # Handle columns that might be mixed int/string/null
                for col in df.columns:
                    if df[col].dtype == pl.Null:
                        # Try to infer type from non-null values
                        non_null = df.filter(pl.col(col).is_not_null())
                        if len(non_null) > 0:
                            # Sample a few non-null values to infer type
                            sample = non_null[col].head(100)
                            # Try to convert to int, then float, then keep as string
                            try:
                                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
                            except:
                                try:
                                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                                except:
                                    df = df.with_columns(pl.col(col).cast(pl.String, strict=False))
            
            # Write to Parquet
            logger.info(f"Writing {len(df)} records to Parquet...")
            df.write_parquet(self.output_path, compression="snappy")
            logger.info(f"✅ Successfully converted {len(df)} records to {self.output_path}")

            # Keep JSONL as backup (optional - could delete)
            # self.jsonl_path.unlink()

            return self.output_path

        except Exception as e:
            logger.error(f"Error converting JSONL to Parquet: {e}")
            logger.info(f"JSONL file preserved at {self.jsonl_path} for recovery")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def get_current_count(self) -> int:
        """Get total number of records written so far."""
        if self.jsonl_path.exists():
            # Count lines in JSONL file
            with open(self.jsonl_path, "r") as f:
                count = sum(1 for line in f if line.strip())
            return count + len(self.buffer)
        return len(self.buffer)

    def exists(self) -> bool:
        """Check if output file already exists."""
        return self.output_path.exists() or self.jsonl_path.exists()

