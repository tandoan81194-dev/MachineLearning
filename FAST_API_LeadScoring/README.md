# Training Lead Scoring API

Du an 2 + 3: train ML model va trien khai thanh API Python cho bai toan cham diem lead/hoc vien trong doanh nghiep dao tao, HRTech hoac CRM.

## Boi canh nghiep vu

Mot cong ty dao tao co nhieu hoc vien/lead dang theo hoc cac khoa data/AI. Doi sales va academic advisor muon biet:

- lead nao co kha nang outcome tot de uu tien tu van goi nang cao
- hoc vien nao can duoc cham soc them
- CRM/LMS co the goi API de lay score tu dong
- manager co the theo doi chat luong pipeline tu van va dao tao

Model du doan xac suat `employed = 1` dua tren thong tin thanh pho, hoc van, kinh nghiem, so gio training va city development index.

## Kien truc

```text
Raw Excel/CSV
  -> Python data preparation
  -> processed_dataset.csv
  -> scikit-learn training pipeline
  -> model.joblib + metrics.json + schema.json
  -> FastAPI inference service
  -> optional S3 archive for uploaded Excel files
  -> optional PostgreSQL persistence for analysis runs
  -> CRM/LMS/Sales dashboard
```

## Cau truc project

```text
training-lead-scoring-api/
├── data/
│   ├── processed/
│   └── samples/
├── models/
├── reports/
├── scripts/
│   ├── evaluate_model.py
│   ├── predict_sample.py
│   ├── prepare_data.py
│   └── train_model.py
├── app/
│   ├── main.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── inference.py
│   ├── schemas.py
│   └── train.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Cai dat

```bash
cd training-lead-scoring-api
python3 -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt
```

Neu ban chi muon chay phan ML trong moi truong hien tai, can co `pandas`, `scikit-learn`, `joblib`, `pydantic`.

## Chay pipeline ML

Du lieu raw mac dinh duoc doc tu:

```text
../training-course-data/
```

Chuan bi du lieu:

```bash
python scripts/prepare_data.py
```

Train model:

```bash
python scripts/train_model.py
```

Evaluate lai model:

```bash
python scripts/evaluate_model.py
```

Predict mot lead mau:

```bash
python scripts/predict_sample.py --input-json data/samples/sample_request.json
```

Chay API:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Sau do mo:

```text
http://127.0.0.1:8080/docs
```

## Chay giao dien Next.js tach rieng

Giao dien khong nam trong backend project. Dashboard duoc tach thanh sibling project:

```text
../training-lead-scoring-dashboard/
```

Neu chay bang Docker Compose, dung file o folder cha:

```text
../docker-compose.yml
```

Terminal 1: chay FastAPI backend.

```bash
cd training-lead-scoring-api
source myenv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Terminal 2: chay Next.js dashboard.

```bash
cd ../training-lead-scoring-dashboard
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Mo dashboard:

```text
http://127.0.0.1:3000
```

Neu backend khong chay o port `8001`, truyen bien moi truong cho Next:

```bash
FASTAPI_BASE_URL=http://127.0.0.1:8080 npm run dev -- --hostname 127.0.0.1 --port 3000
```

Dashboard hien co:

- Score mot lead bang form
- Lead Queue de sap xep uu tien
- Excel Analysis de upload unseen employee file va suy ra Lead Queue
- Model Health de xem status, metrics va feature schema

## PostgreSQL persistence

Endpoint `POST /analysis-runs` co the luu thong tin nhan vien va ket qua scoring vao PostgreSQL.

Khi chua co database, khong can cau hinh gi them. API van score binh thuong va tra ve:

```json
{
  "saved_to_database": false,
  "persistence_status": "DATABASE_URL is not configured; analysis was not persisted."
}
```

Khi co database, set bien moi truong:

```bash
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_USER="postgres"
export DB_PASS="password"
export DB_DATABASE="lead_scoring"
```

API se tu tao 2 bang toi thieu neu chua co:

- `analysis_runs`
- `employee_lead_scores`

Neu muon dung connection string truc tiep, co the set `DATABASE_URL`; bien nay se duoc uu tien hon cac bien `DB_*`.

## AWS S3 source file archive

Endpoint `POST /source-files` nhan raw Excel bytes tu dashboard va upload file goc len S3 neu da cau hinh trong backend env:

```bash
cp .env.example .env
```

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=your-bucket-name
S3_PREFIX=lead-scoring/source-files
```

Neu `S3_BUCKET_NAME` chua duoc set, endpoint se tra ve `saved_to_s3=false` va app van tiep tuc scoring binh thuong. Sau khi upload S3 thanh cong, dashboard gui metadata `source_s3_bucket`, `source_s3_key`, `source_s3_uri`, `source_s3_url` vao `POST /analysis-runs` de luu cung analysis run.

## API contract

### `GET /health`

Kiem tra service co san sang khong.

### `GET /model-info`

Xem model version, feature list va duong dan artifact.

### `GET /model-metrics`

Xem metrics baseline tu file `models/metrics.json`.

### `POST /predict`

Input:

```json
{
  "city": "city_103",
  "gender": "Male",
  "enrolled_university": "no_enrollment",
  "education_level": "Graduate",
  "major_discipline": "STEM",
  "relevent_experience": "Has relevent experience",
  "experience": ">20",
  "company_size": "50-99",
  "company_type": "Pvt Ltd",
  "last_new_job": "1",
  "training_hours": 36,
  "city_development_index": 0.92
}
```

Output:

```json
{
  "employment_probability": 0.74,
  "segment": "high_potential",
  "recommended_action": "Assign senior counselor and offer advanced career track",
  "model_version": "baseline-2026-06"
}
```

### `POST /batch-predict`

Nhan danh sach lead va tra ve danh sach score.

### `POST /analysis-runs`

Nhan danh sach nhan vien kem metadata, score bang model hien tai, va neu `DATABASE_URL` duoc cau hinh thi luu ca employee input lan prediction vao PostgreSQL.

Input:

```json
{
  "source_filename": "unseen_employee_leads_20.xlsx",
  "source_sheet": "Unseen Employees",
  "employees": [
    {
      "employee_id": "EMP-0001",
      "employee_name": "Nguyen Minh Anh",
      "department": "Data",
      "role": "Senior Data Analyst",
      "manager": "Tran Quang Huy",
      "scenario_note": "Senior STEM profile",
      "source_row": 5,
      "lead": {
        "city": "city_103",
        "gender": "Female",
        "enrolled_university": "no_enrollment",
        "education_level": "Masters",
        "major_discipline": "STEM",
        "relevent_experience": "Has relevent experience",
        "experience": ">20",
        "company_size": "10000+",
        "company_type": "Pvt Ltd",
        "last_new_job": ">4",
        "training_hours": 18,
        "city_development_index": 0.92
      }
    }
  ]
}
```

Output:

```json
{
  "analysis_run_id": "uuid",
  "saved_to_database": true,
  "persistence_status": "Analysis run and employee predictions were persisted.",
  "results": []
}
```

## Quy trinh du an thuc te

1. Chot business objective: score lead de ho tro CRM/Sales.
2. Chuan hoa raw data: merge Excel/CSV bang `enrollee_id` va `city`.
3. Tao baseline model: Logistic Regression voi preprocessing pipeline.
4. Danh gia: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
5. Dong goi artifact: `model.joblib`, `metrics.json`, `schema.json`.
6. Expose API: FastAPI endpoints cho online scoring va batch scoring.
7. Viet README va sample request de team khac tich hop.

## Mo rong cho cac du an sau

- Docker: dong goi service thanh container.
- AWS: deploy API len EC2, model artifact co the dua len S3.
- CI/CD: chay test va deploy bang GitHub Actions.
- MLflow: log experiments, metrics va model registry.
- DVC: version raw/processed dataset.
- Airflow: retrain theo lich.
- Prometheus/Grafana: monitor latency, error rate, prediction distribution.
