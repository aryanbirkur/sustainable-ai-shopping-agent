# Sustainable AI Shopping Agent

An AI-powered sustainable product recommendation platform, built in
milestones. See `docs/` for the full architecture (Milestone 1) and
dataset documentation (Milestone 2).

## Status
- Milestone 1: Project foundation — DONE
- Milestone 2: Dataset + data pipeline — DONE
- Milestone 3: Sustainability intelligence engine — NOT STARTED

## Quickstart (Milestone 2)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/generate_synthetic_data.py
python scripts/run_pipeline.py
python scripts/data_quality_report.py
python scripts/eda.py
pytest tests/ -v
```

See `docs/dataset_sourcing.md` for why the current dataset is synthetic,
and `docs/data_dictionary.md` for the full field-by-field schema.
