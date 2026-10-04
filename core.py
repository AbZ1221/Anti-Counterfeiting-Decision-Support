# === Type annotations ===
# Postpone evaluation of type hints until they are needed.
from __future__ import annotations

# === Standard library: database access and result containers ===
# Serialize assessment snapshots to JSON for storage with analyst reviews.
import json
# Read and write the local SQLite database containing review history.
import sqlite3
# Close database connections when their with block finishes, even on errors.
from contextlib import closing
# dataclass builds the assessment container; asdict converts it into a dictionary.
from dataclasses import dataclass, asdict
# Generate timezone-aware UTC timestamps for saved reviews.
from datetime import datetime, timezone
# Construct paths relative to this module's directory.
from pathlib import Path

# === Data processing and machine learning ===
# Clamp numerical risk signals to the range [0, 1].
import numpy as np
# Load listings, clean numeric columns and assemble tabular assessment results.
import pandas as pd
# Convert listing text into weighted word and two-word features (TF-IDF).
from sklearn.feature_extraction.text import TfidfVectorizer
# Learn a binary text classifier from confirmed genuine and counterfeit examples.
from sklearn.linear_model import LogisticRegression
# Chain text feature extraction and classification into one trainable model.
from sklearn.pipeline import Pipeline
# Compare listing images with verified references and return visual evidence.
from vision import AdvancedVisionEngine

# === Project paths and seller-link fields ===
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "listings.csv"
DB_PATH = ROOT / "investigations.db"

# Shared values in these fields establish direct connections between seller accounts.
IDENTIFIERS = ["phone", "email", "address", "image_hash", "payment_ref"]


@dataclass
class Assessment:
    # Individual risks use [0, 1]; overall_risk uses [0, 100].
    # Keep the scores together with visual evidence and human-readable explanations.
    listing_id: str
    visual_risk: float
    vision_details: dict
    text_risk: float
    price_risk: float
    behaviour_risk: float
    network_risk: float
    history_risk: float
    overall_risk: float
    risk_level: str
    reasons: list[str]

    def as_dict(self):
        # Produce a plain dictionary for tables, JSON snapshots and reports.
        return asdict(self)


# === Listing preparation and text model training ===
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    # Replace missing cells, then coerce invalid numeric values to zero.
    df = pd.read_csv(path).fillna("")
    numeric = ["price", "market_price", "account_age_days", "listing_count_30d", "prior_confirmed_cases"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def train_text_model(df: pd.DataFrame) -> Pipeline:
    # Only confirmed labels are training targets; unreviewed listings are excluded.
    training = df[df["label"].isin(["confirmed_genuine", "confirmed_counterfeit"])].copy()
    if training["label"].nunique() < 2:
        raise ValueError("Training data requires authoritative examples from both classes.")
    # Combine title and description; counterfeit is class 1 and genuine is class 0.
    text = training["title"].astype(str) + " " + training["description"].astype(str)
    y = (training["label"] == "confirmed_counterfeit").astype(int)
    # Use individual words and word pairs, with logarithmic term-frequency scaling.
    # Balanced class weights compensate for unequal numbers of training examples.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    model.fit(text, y)
    return model


# === Seller connections through shared identifiers ===
def build_entity_graph(df: pd.DataFrame) -> dict:
    # Store both directions: seller -> identifiers and identifier -> sellers.
    graph = {"seller_entities": {}, "entity_sellers": {}}
    for _, row in df.iterrows():
        seller = str(row["seller_id"])
        graph["seller_entities"].setdefault(seller, set())
        for field in IDENTIFIERS:
            # Normalize case/whitespace and ignore missing or placeholder identifiers.
            value = str(row[field]).strip().lower()
            if value and value not in {"unknown", "nan"}:
                # Include the field name so values from different identifier types stay distinct.
                entity = f"{field}:{value}"
                graph["seller_entities"][seller].add(entity)
                graph["entity_sellers"].setdefault(entity, set()).add(seller)
    return graph


def linked_sellers(graph: dict, seller_id: str) -> list[dict]:
    # Find direct links only, returning one result per other seller/shared identifier pair.
    # A seller sharing several identifiers can therefore appear more than once.
    if seller_id not in graph["seller_entities"]:
        return []
    results = []
    seen = set()
    for entity in graph["seller_entities"][seller_id]:
        for target in graph["entity_sellers"].get(entity, set()):
            key = (target, entity)
            # Exclude self-links and duplicate pairs.
            if target != seller_id and key not in seen:
                results.append({"seller_id": target, "shared_entity": entity})
                seen.add(key)
    return results


# === Risk calculation and supporting explanations ===
def _clip(value: float) -> float:
    # Keep heuristic component scores within the common [0, 1] range.
    return float(np.clip(value, 0, 1))


def assess_listing(row: pd.Series, model: Pipeline, graph: dict, vision_engine: AdvancedVisionEngine) -> Assessment:
    # Compare the image with its reference and check for expected brand-related text.
    vision = vision_engine.analyse(
        str(row["listing_id"]), str(row["image_path"]), str(row["reference_path"]),
        f"{row['brand']} authentic reference batch",
    )
    text = f"{row['title']} {row['description']}"
    # Column 1 is the classifier's estimated probability for the counterfeit text class.
    text_risk = float(model.predict_proba([text])[0, 1])

    # Discounts up to 20% score zero; discounts of 85% or more reach maximum price risk.
    # A missing/nonpositive market price disables this signal to avoid division by zero.
    discount = 0 if row["market_price"] <= 0 else 1 - row["price"] / row["market_price"]
    price_risk = _clip((discount - 0.20) / 0.65)

    # Volume rises from zero at 30 listings to one at 200 listings in 30 days.
    # Account-age risk decreases to zero once an account reaches 90 days old.
    volume = _clip((row["listing_count_30d"] - 30) / 170)
    new_account = _clip((90 - row["account_age_days"]) / 90)
    behaviour_risk = _clip(0.60 * volume + 0.40 * new_account)

    links = linked_sellers(graph, str(row["seller_id"]))
    # Count seller/identifier pairs, not unique sellers; three pairs saturate this signal.
    network_risk = _clip(len(links) / 3)
    # Two or more prior confirmed cases produce the maximum history score.
    history_risk = _clip(float(row["prior_confirmed_cases"]) / 2)

    # Combine six signals with fixed weights and express risk on a 100-point scale.
    # This weighted heuristic is not a calibrated probability of counterfeiting.
    overall = 100 * (
        0.35 * vision.visual_risk + 0.20 * text_risk + 0.18 * price_risk
        + 0.12 * behaviour_risk + 0.10 * network_risk + 0.05 * history_risk
    )
    # Demonstration thresholds only; calibrate these against validation data and review capacity.
    level = "High" if overall >= 60 else "Medium" if overall >= 35 else "Low"

    # Add threshold-based explanations alongside the vision engine's findings.
    reasons = [f"Visual: {reason}" for reason in vision.reasons]
    if text_risk >= 0.60:
        reasons.append("Listing language resembles previously verified counterfeit examples.")
    if discount >= 0.50:
        reasons.append(f"Price is {discount:.0%} below the reference market price.")
    if row["account_age_days"] < 30:
        reasons.append("Seller account is less than 30 days old.")
    if row["listing_count_30d"] > 100:
        reasons.append("Seller posted more than 100 listings in the last 30 days.")
    if links:
        reasons.append(f"Seller shares identifiers with {len(links)} other account(s).")
    if row["prior_confirmed_cases"] > 0:
        reasons.append("Seller has an authorised historical case indicator in the sample data.")
    if not reasons:
        reasons.append("No strong indicator was detected; routine monitoring may be sufficient.")

    # Round stored scores for display while retaining detailed visual evidence.
    return Assessment(
        listing_id=str(row["listing_id"]), visual_risk=vision.visual_risk,
        vision_details=vision.as_dict(), text_risk=round(text_risk, 4),
        price_risk=round(price_risk, 4), behaviour_risk=round(behaviour_risk, 4),
        network_risk=round(network_risk, 4), history_risk=round(history_risk, 4),
        overall_risk=round(overall, 1), risk_level=level, reasons=reasons,
    )


def assess_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, Pipeline]:
    # Reuse one text model, entity graph and vision engine across all listings.
    model = train_text_model(df)
    graph = build_entity_graph(df)
    vision_engine = AdvancedVisionEngine()
    assessments = pd.DataFrame([assess_listing(row, model, graph, vision_engine).as_dict() for _, row in df.iterrows()])
    # Attach scores to the original listing columns and expose reusable graph/model objects.
    return df.merge(assessments, on="listing_id"), graph, model


# === Persistent human reviews ===
def init_db(path: Path = DB_PATH):
    # Create the review table if needed without clearing existing review records.
    with closing(sqlite3.connect(path)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
              id INTEGER PRIMARY KEY AUTOINCREMENT, listing_id TEXT NOT NULL,
              analyst TEXT NOT NULL, decision TEXT NOT NULL, notes TEXT,
              created_at TEXT NOT NULL, assessment_json TEXT NOT NULL
            )
        """)
        con.commit()


def save_review(listing_id: str, analyst: str, decision: str, notes: str, assessment: dict, path: Path = DB_PATH):
    # Append a review with a UTC timestamp and a snapshot of the assessment at save time.
    init_db(path)
    with closing(sqlite3.connect(path)) as con:
        # Parameter placeholders bind values safely instead of interpolating them into SQL.
        con.execute(
            "INSERT INTO reviews(listing_id, analyst, decision, notes, created_at, assessment_json) VALUES(?,?,?,?,?,?)",
            (listing_id, analyst, decision, notes, datetime.now(timezone.utc).isoformat(), json.dumps(assessment)),
        )
        con.commit()


def get_reviews(listing_id: str, path: Path = DB_PATH) -> pd.DataFrame:
    # Return this listing's review history, with the most recently inserted review first.
    init_db(path)
    with closing(sqlite3.connect(path)) as con:
        return pd.read_sql_query(
            "SELECT analyst, decision, notes, created_at FROM reviews WHERE listing_id=? ORDER BY id DESC",
            con, params=(listing_id,),
        )
