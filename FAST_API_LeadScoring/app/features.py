from __future__ import annotations


TARGET_COLUMN = "employed"
ID_COLUMN = "enrollee_id"

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "relevent_experience",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
]

NUMERIC_FEATURES = [
    "training_hours",
    "city_development_index",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

OUTPUT_COLUMNS = [
    ID_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]
