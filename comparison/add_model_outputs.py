"""Attach VADER and FinBERT outputs to the hand-labelled 200-post sample.

    python -m comparison.add_model_outputs
    -> comparison/labelling_200_with_models.csv

Scores are read from what the pipeline already stored, so the comparison is
against exactly the numbers used downstream:

* VADER compound: posts.sentiment in data/goldpulse.sqlite3, labelled with the
  project's usual +/-0.05 neutral band (processing/bucket.py).
* FinBERT: data/clean/post_sentiment_finbert.parquet (sentiment/finbert.py).

vader_text_mismatch flags posts where re-scoring the CSV's text with VADER does
not reproduce the stored score, i.e. the labelled text differs from the text
the collector saw.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from processing.bucket import NEUTRAL_BAND
from sentiment.finbert import COLUMNS as FINBERT_COLUMNS, OUTPUT_PATH as FINBERT_PATH
from sentiment.scorer import clean_text, score_sentiment

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "labelling_200_flagged.csv"
OUTPUT_PATH = HERE / "labelling_200_with_models.csv"


def vader_label(score: pd.Series) -> np.ndarray:
    return np.where(score > NEUTRAL_BAND, "bullish",
                    np.where(score < -NEUTRAL_BAND, "bearish", "neutral"))


def main() -> int:
    labels = pd.read_csv(INPUT_PATH)

    with sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
        vader = pd.read_sql("SELECT post_id, sentiment AS vader_compound FROM posts", conn)
    finbert = pd.read_parquet(FINBERT_PATH)[["post_id", *FINBERT_COLUMNS]]

    out = labels.merge(vader, on="post_id", how="left", validate="1:1")
    out["vader_label"] = vader_label(out["vader_compound"])
    out = out.merge(finbert, on="post_id", how="left", validate="1:1")

    missing = out["vader_compound"].isna() | out["finbert_label"].isna()
    if missing.any():
        raise ValueError(f"no stored score for: {out.loc[missing, 'post_id'].tolist()}")

    rescored = out["text"].map(lambda t: score_sentiment(clean_text(t)))
    out["vader_text_mismatch"] = (rescored - out["vader_compound"]).abs() > 1e-4

    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(out)} posts, "
          f"{int(out['vader_text_mismatch'].sum())} with text differing from the scored text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
