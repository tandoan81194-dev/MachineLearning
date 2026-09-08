from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_PATH, SAMPLES_DIR
from app.inference import LeadScoringService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local prediction from a JSON payload.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--input-json", default=str(SAMPLES_DIR / "sample_request.json"))
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    prediction = LeadScoringService(Path(args.model)).predict_one(payload)
    print(prediction.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
