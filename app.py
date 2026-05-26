import os
import re
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"

DATABASE = "house.db"
MODEL_PATH = "best_model02.pkl"
USD_TO_INR_RATE = 94.9432
BRAND_NAME = "Northstone"

# The model was trained with this exact column order. Keep this list as the
# single source of truth for form parsing, dataframe creation, and DB columns.
FEATURES = [
    "id",
    "date",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

INT_FIELDS = {
    "id",
    "date",
    "bedrooms",
    "sqft_living",
    "sqft_lot",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "sqft_living15",
    "sqft_lot15",
}

FLOAT_FIELDS = {"bathrooms", "floors", "lat", "long"}

FIELD_LABELS = {
    "id": "House ID",
    "date": "Sale Date",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "sqft_living": "Sqft Living",
    "sqft_lot": "Sqft Lot",
    "floors": "Floors",
    "waterfront": "Waterfront",
    "view": "View",
    "condition": "Condition",
    "grade": "Grade",
    "sqft_above": "Sqft Above",
    "sqft_basement": "Sqft Basement",
    "yr_built": "Year Built",
    "yr_renovated": "Year Renovated",
    "zipcode": "Zipcode",
    "lat": "Latitude",
    "long": "Longitude",
    "sqft_living15": "Sqft Living 15",
    "sqft_lot15": "Sqft Lot 15",
}

DEFAULT_VALUES = {
    "id": "7129300520",
    "date": "2014-10-13",
    "bedrooms": "3",
    "bathrooms": "1.0",
    "sqft_living": "1180",
    "sqft_lot": "5650",
    "floors": "1.0",
    "waterfront": "0",
    "view": "0",
    "condition": "3",
    "grade": "7",
    "sqft_above": "1180",
    "sqft_basement": "0",
    "yr_built": "1955",
    "yr_renovated": "0",
    "zipcode": "98178",
    "lat": "47.5112",
    "long": "-122.257",
    "sqft_living15": "1340",
    "sqft_lot15": "5650",
    "place_name": "Downtown Seattle",
}

LOCATION_OPTIONS = [
    {
        "city": "Seattle",
        "area": "Rainier Valley",
        "zipcode": "98178",
        "lat": "47.5112",
        "long": "-122.257",
    },
    {
        "city": "Bellevue",
        "area": "Downtown Bellevue",
        "zipcode": "98004",
        "lat": "47.6101",
        "long": "-122.2015",
    },
    {
        "city": "Redmond",
        "area": "Overlake",
        "zipcode": "98052",
        "lat": "47.6718",
        "long": "-122.1232",
    },
    {
        "city": "Kirkland",
        "area": "Juanita",
        "zipcode": "98034",
        "lat": "47.7170",
        "long": "-122.2050",
    },
    {
        "city": "Tacoma",
        "area": "North End",
        "zipcode": "98406",
        "lat": "47.2630",
        "long": "-122.4880",
    },
    {
        "city": "Renton",
        "area": "Highlands",
        "zipcode": "98056",
        "lat": "47.5050",
        "long": "-122.1810",
    },
]

FIELD_TYPES = {
    "id": "number",
    "date": "date",
    "bedrooms": "number",
    "bathrooms": "number",
    "sqft_living": "number",
    "sqft_lot": "number",
    "floors": "number",
    "waterfront": "number",
    "view": "number",
    "condition": "number",
    "grade": "number",
    "sqft_above": "number",
    "sqft_basement": "number",
    "yr_built": "number",
    "yr_renovated": "number",
    "zipcode": "number",
    "lat": "number",
    "long": "number",
    "sqft_living15": "number",
    "sqft_lot15": "number",
}

DB_FEATURE_COLUMNS = {feature: ("house_id" if feature == "id" else feature) for feature in FEATURES}

STEP_VALUES = {
    "bathrooms": "0.25",
    "floors": "0.5",
    "lat": "any",
    "long": "any",
}


def get_model_feature_names():
    """Return the saved LightGBM feature names when available."""
    booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
    if booster is not None:
        try:
            return booster.feature_name()
        except Exception:
            return FEATURES
    return FEATURES


def get_sale_year(features):
    return int(str(features["date"])[:4])


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_model_input(features):
    """Build the prediction dataframe expected by the saved model.

    The UI stores the original dataset fields. Some saved LightGBM models are
    trained after feature engineering, so this function maps raw form data into
    the actual feature names recorded inside best_model02.pkl.
    """
    model_features = get_model_feature_names()
    sale_year = get_sale_year(features)
    yr_renovated = features["yr_renovated"]
    renovated = 1 if yr_renovated > 0 else 0

    available_values = {
        **features,
        "house_age": sale_year - features["yr_built"],
        "renovation_age": sale_year - yr_renovated if renovated else 0,
        "total_sqft": features["sqft_above"] + features["sqft_basement"],
        "bath_per_bed": safe_divide(features["bathrooms"], features["bedrooms"]),
        "living_lot_ratio": safe_divide(features["sqft_living"], features["sqft_lot"]),
        "living_vs_neighbour": safe_divide(
            features["sqft_living"], features["sqft_living15"]
        ),
        "renovated": renovated,
    }

    missing = [name for name in model_features if name not in available_values]
    if missing:
        raise ValueError(
            "The saved model expects unsupported feature(s): " + ", ".join(missing)
        )

    input_df = pd.DataFrame(
        [[available_values[name] for name in model_features]], columns=model_features
    )

    # Preserve LightGBM's saved pandas categorical metadata. In this model,
    # zipcode was trained as a categorical feature.
    booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
    if booster is not None and getattr(booster, "pandas_categorical", None):
        categorical_columns = [
            name
            for name in model_features
            if name in {"zipcode"} and booster.pandas_categorical
        ]
        for index, column in enumerate(categorical_columns):
            if index < len(booster.pandas_categorical):
                input_df[column] = pd.Categorical(
                    input_df[column], categories=booster.pandas_categorical[index]
                )

    return input_df


def get_model():
    """Load the LightGBM model saved with joblib."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. Place it beside app.py."
        )
    return joblib.load(MODEL_PATH)


model = get_model()


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    columns = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "house_id INTEGER NOT NULL",
        "date INTEGER NOT NULL",
        "bedrooms INTEGER NOT NULL",
        "bathrooms REAL NOT NULL",
        "sqft_living INTEGER NOT NULL",
        "sqft_lot INTEGER NOT NULL",
        "floors REAL NOT NULL",
        "waterfront INTEGER NOT NULL",
        "view INTEGER NOT NULL",
        "condition INTEGER NOT NULL",
        "grade INTEGER NOT NULL",
        "sqft_above INTEGER NOT NULL",
        "sqft_basement INTEGER NOT NULL",
        "yr_built INTEGER NOT NULL",
        "yr_renovated INTEGER NOT NULL",
        "zipcode INTEGER NOT NULL",
        "lat REAL NOT NULL",
        "long REAL NOT NULL",
        "sqft_living15 INTEGER NOT NULL",
        "sqft_lot15 INTEGER NOT NULL",
        "place_name TEXT",
        "predicted_price REAL NOT NULL",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]

    with get_db_connection() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS predictions ({', '.join(columns)})")
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(predictions)")
        }
        if "place_name" not in existing_columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN place_name TEXT")
        conn.commit()


def parse_date_to_int(value):
    """Convert date input such as 20141013T000000 or 2014-10-13 to 20141013."""
    if value is None:
        raise ValueError("date is required")

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("date is required")

    digits = re.sub(r"\D", "", raw_value)
    if len(digits) < 8:
        raise ValueError("date must contain year, month, and day")

    date_digits = digits[:8]
    datetime.strptime(date_digits, "%Y%m%d")
    return int(date_digits)


def display_date(value):
    """Return YYYY-MM-DD for HTML date inputs from stored YYYYMMDD integers."""
    value = str(value)
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def get_location_by_zipcode(zipcode):
    zipcode = str(zipcode)
    for location in LOCATION_OPTIONS:
        if location["zipcode"] == zipcode:
            return location
    return None


def sanitize_place_name(value):
    return str(value or "").strip()[:120]


def location_label_from_form(form_data, features=None):
    city = str(form_data.get("city", "")).strip()
    place_name = sanitize_place_name(form_data.get("place_name"))

    if not city and features:
        location = get_location_by_zipcode(features.get("zipcode"))
        if location:
            city = location["city"]
            place_name = place_name or location["area"]

    parts = [part for part in [place_name, city] if part]
    return ", ".join(parts) if parts else "Selected market"


def display_location_from_record(record):
    place_name = sanitize_place_name(record["place_name"] if "place_name" in record.keys() else "")
    location = get_location_by_zipcode(record["zipcode"])
    if place_name and location:
        return f"{place_name}, {location['city']}"
    if place_name:
        return place_name
    if location:
        return f"{location['area']}, {location['city']}"
    return f"Zipcode {record['zipcode']}"


def preprocess_form(form_data):
    """Convert form strings to the numeric types expected by the trained model."""
    processed = {}

    for feature in FEATURES:
        raw_value = form_data.get(feature)

        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(f"{FIELD_LABELS[feature]} is required")

        try:
            if feature == "date":
                processed[feature] = parse_date_to_int(raw_value)
            elif feature in INT_FIELDS:
                processed[feature] = int(float(raw_value))
            elif feature in FLOAT_FIELDS:
                processed[feature] = float(raw_value)
            else:
                processed[feature] = raw_value
        except ValueError as exc:
            if feature == "date":
                raise ValueError(f"Invalid {FIELD_LABELS[feature]}: {exc}") from exc
            raise ValueError(
                f"{FIELD_LABELS[feature]} must be a valid number"
            ) from exc

    return processed


def predict_price(features):
    """Create model input and return the predicted price in Indian rupees."""
    input_df = build_model_input(features)

    try:
        prediction_usd = model.predict(input_df)[0]
    except AttributeError as exc:
        # If scikit-learn is missing, LightGBM's sklearn wrapper can fail with:
        # "'super' object has no attribute 'get_params'". The saved model still
        # contains the underlying LightGBM Booster, so use it directly.
        if "get_params" not in str(exc) or not hasattr(model, "booster_"):
            raise
        prediction_usd = model.booster_.predict(input_df)[0]

    return float(prediction_usd) * USD_TO_INR_RATE


@app.template_filter("inr")
def format_inr(value):
    """Format a number with Indian comma grouping and an INR label."""
    amount = float(value)
    sign = "-" if amount < 0 else ""
    whole, decimal = f"{abs(amount):.2f}".split(".")

    if len(whole) > 3:
        last_three = whole[-3:]
        leading = whole[:-3]
        groups = []
        while len(leading) > 2:
            groups.insert(0, leading[-2:])
            leading = leading[:-2]
        if leading:
            groups.insert(0, leading)
        whole = ",".join(groups + [last_three])

    return f"{sign}INR {whole}.{decimal}"


def insert_prediction(features, predicted_price, place_name=""):
    columns = [DB_FEATURE_COLUMNS[name] for name in FEATURES] + [
        "place_name",
        "predicted_price",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})"
    values = [features[name] for name in FEATURES] + [
        sanitize_place_name(place_name),
        predicted_price,
    ]

    with get_db_connection() as conn:
        cursor = conn.execute(sql, values)
        conn.commit()
        return cursor.lastrowid


def get_prediction_record(record_id):
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()


def update_prediction(record_id, features, predicted_price, place_name=""):
    assignments = ", ".join([f"{DB_FEATURE_COLUMNS[name]} = ?" for name in FEATURES])
    sql = (
        f"UPDATE predictions SET {assignments}, place_name = ?, predicted_price = ?, "
        "created_at = CURRENT_TIMESTAMP WHERE id = ?"
    )
    values = [features[name] for name in FEATURES] + [
        sanitize_place_name(place_name),
        predicted_price,
        record_id,
    ]

    with get_db_connection() as conn:
        conn.execute(sql, values)
        conn.commit()


def delete_prediction(record_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        conn.commit()


def template_context(**extra):
    context = {
        "brand_name": BRAND_NAME,
        "features": FEATURES,
        "field_labels": FIELD_LABELS,
        "field_types": FIELD_TYPES,
        "step_values": STEP_VALUES,
        "default_values": DEFAULT_VALUES,
        "db_feature_columns": DB_FEATURE_COLUMNS,
        "location_options": LOCATION_OPTIONS,
    }
    context.update(extra)
    return context


@app.route("/")
def index():
    return render_template("index.html", **template_context(form_values=DEFAULT_VALUES))


@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    try:
        features = preprocess_form(request.form)
        predicted_price = predict_price(features)
        place_name = sanitize_place_name(request.form.get("place_name"))
        record_id = insert_prediction(features, predicted_price, place_name)
        return render_template(
            "result.html",
            **template_context(
                predicted_price=predicted_price,
                record_id=record_id,
                location_label=location_label_from_form(request.form, features),
                property_summary=features,
                place_name=place_name,
            ),
        )
    except Exception as exc:
        flash(str(exc), "error")
        return render_template("index.html", **template_context(form_values=form_values))


@app.route("/records")
def records():
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return render_template(
        "records.html",
        **template_context(
            records=rows,
            display_date=display_date,
            display_location=display_location_from_record,
        ),
    )


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete(record_id):
    try:
        delete_prediction(record_id)
        flash("Prediction record deleted.", "success")
    except Exception as exc:
        flash(f"Could not delete record: {exc}", "error")
    return redirect(url_for("records"))


@app.route("/update/<int:record_id>", methods=["GET", "POST"])
def update(record_id):
    record = get_prediction_record(record_id)
    if record is None:
        flash("Prediction record not found.", "error")
        return redirect(url_for("records"))

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            features = preprocess_form(request.form)
            predicted_price = predict_price(features)
            place_name = sanitize_place_name(request.form.get("place_name"))
            update_prediction(record_id, features, predicted_price, place_name)
            flash("Prediction record updated.", "success")
            return render_template(
                "result.html",
                **template_context(
                    predicted_price=predicted_price,
                    record_id=record_id,
                    location_label=location_label_from_form(request.form, features),
                    property_summary=features,
                    place_name=place_name,
                ),
            )
        except Exception as exc:
            flash(str(exc), "error")
            return render_template(
                "update.html",
                **template_context(record_id=record_id, form_values=form_values),
            )

    form_values = {feature: record[DB_FEATURE_COLUMNS[feature]] for feature in FEATURES}
    form_values["date"] = display_date(form_values["date"])
    form_values["place_name"] = record["place_name"] or ""
    location = get_location_by_zipcode(form_values["zipcode"])
    if location:
        form_values["city"] = location["city"]
    return render_template(
        "update.html", **template_context(record_id=record_id, form_values=form_values)
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)
