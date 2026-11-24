# VoltCast — Electricity Demand Forecasting & Load Profiling

End-to-end undergraduate ML project using the UCI Individual Household Electric Power Consumption dataset.
See [REPORT.md](REPORT.md) for results and [ARCHITECTURE.md](ARCHITECTURE.md) for data contracts.

## Setup

Python 3.11 or 3.12.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Linux / macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running

```bash
python -m voltcast.cli fetch-data      # downloads the UCI zip into data/raw/
python -m voltcast.cli run             # full pipeline -> artifacts/full/
python -m voltcast.cli smoke           # synthetic 35-day fixture -> artifacts/smoke/
python -m pytest -q                    # tests
streamlit run dashboard/app.py         # dashboard (pick "full" in the sidebar)
```

Individual stages: `data`, `features`, `regression`, `classification`, `pca`, `profiles`,
`profile-forecast`, `circuits` (add `--smoke` to run against the smoke artifacts).
