from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_RAW_DATA_DIR, PROCESSED_DATA_PATH
from app.data import save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw enrollment data for model training.")
    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--output", default=str(PROCESSED_DATA_PATH))
    args = parser.parse_args()

    dataset, report = save_dataset(
        output_path=Path(args.output),
        raw_data_dir=Path(args.raw_data_dir),
    )

    print(f"Saved processed dataset: {Path(args.output).resolve()}")
    print(f"Rows: {report.rows}")
    print(f"Columns: {report.columns}")
    print(f"Duplicate IDs: {report.duplicate_ids}")
    print(f"Target positive rate: {report.target_positive_rate:.4f}")
    if report.missing_cells:
        print("Missing cells:")
        for column, count in report.missing_cells.items():
            print(f"  - {column}: {count}")
    print(f"Preview rows: {min(5, len(dataset))}")


if __name__ == "__main__":
    main()
