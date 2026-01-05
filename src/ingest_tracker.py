"""Progress tracking and resume capability for data ingestion."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class IngestionTracker:
    """Tracks ingestion progress and enables resume capability."""

    def __init__(self, status_file: Path):
        """Initialize tracker with status file path."""
        self.status_file = status_file
        self.status: dict = {}
        self._load_status()

    def _load_status(self) -> None:
        """Load status from file if it exists."""
        if self.status_file.exists():
            try:
                with open(self.status_file, "r") as f:
                    self.status = json.load(f)
                logger.info(f"Loaded ingestion status from {self.status_file}")
            except Exception as e:
                logger.warning(f"Could not load status file: {e}. Starting fresh.")
                self.status = {}
        else:
            self.status = {}

    def _save_status(self) -> None:
        """Save current status to file."""
        try:
            self.status["last_updated"] = datetime.now().isoformat()
            with open(self.status_file, "w") as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save status file: {e}")

    def start_ingestion(
        self, start_year: int, end_year: int, output_dir: Path, force: bool = False
    ) -> None:
        """Mark ingestion as started."""
        self.status = {
            "start_year": start_year,
            "end_year": end_year,
            "output_dir": str(output_dir),
            "force": force,
            "started_at": datetime.now().isoformat(),
            "completed_steps": [],
            "current_step": None,
            "progress": {},
        }
        self._save_status()
        logger.info(f"Started ingestion tracking: {start_year}-{end_year}")

    def mark_step_started(self, step_name: str) -> None:
        """Mark a step as started."""
        self.status["current_step"] = step_name
        self.status["progress"][step_name] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
        }
        self._save_status()
        logger.info(f"Started step: {step_name}")

    def mark_step_completed(self, step_name: str, result_path: Optional[Path] = None) -> None:
        """Mark a step as completed."""
        if step_name not in self.status["completed_steps"]:
            self.status["completed_steps"].append(step_name)
        
        self.status["progress"][step_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result_path": str(result_path) if result_path else None,
        }
        self.status["current_step"] = None
        self._save_status()
        logger.info(f"Completed step: {step_name}")

    def mark_step_failed(self, step_name: str, error: str) -> None:
        """Mark a step as failed."""
        self.status["progress"][step_name] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": str(error),
        }
        self.status["current_step"] = None
        self._save_status()
        logger.error(f"Step failed: {step_name} - {error}")

    def update_progress(self, step_name: str, progress_info: dict) -> None:
        """Update progress information for a step."""
        if step_name not in self.status["progress"]:
            self.status["progress"][step_name] = {}
        self.status["progress"][step_name].update(progress_info)
        self.status["progress"][step_name]["last_updated"] = datetime.now().isoformat()
        self._save_status()

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step is already completed."""
        return step_name in self.status["completed_steps"]

    def get_progress_summary(self) -> dict:
        """Get summary of current progress."""
        return {
            "completed_steps": self.status.get("completed_steps", []),
            "current_step": self.status.get("current_step"),
            "progress": self.status.get("progress", {}),
        }

    def finish_ingestion(self) -> None:
        """Mark ingestion as finished."""
        self.status["completed_at"] = datetime.now().isoformat()
        self.status["current_step"] = None
        self._save_status()
        logger.info("Ingestion completed successfully")

