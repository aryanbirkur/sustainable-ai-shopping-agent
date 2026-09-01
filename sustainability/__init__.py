"""
sustainability/

Milestone 3: Sustainability Intelligence Engine.

Modules:
    scoring_engine.py         - Rule-based baseline scorer (transparent, no ML).
    ml_scorer.py               - ML (scikit-learn) component trained on a documented
                                  synthetic proxy target; see its docstring for why.
    explanation_generator.py   - Rule-based/template-based human-readable explanations.
    batch_score.py              - Runs the full engine over products_clean.csv and
                                  writes data/processed/products_scored.csv.
"""
