import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

def _cast_to_str(X):
    """Cast all columns to str so SimpleImputer(most_frequent) works with mixed dtypes."""
    import pandas as pd
    df = pd.DataFrame(X)
    for col in df.columns:
        df[col] = df[col].where(df[col].notna(), other=None)
        df[col] = df[col].astype(object)
    return df
SEED = 123
NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_clean",
    "amenity_count",
    "minimum_nights_capped",
    "host_tenure_years",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "room_type",
    "property_type_grouped",
    "host_is_superhost",
    "instant_bookable",
]
TARGET = "price"
def _build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("cast", FunctionTransformer(_cast_to_str)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    lgbm = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgbm),
    ])
@lru_cache(maxsize=1)
def get_model(listings_hash: int):
    from data_loader import get_data
    listings = get_data()
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = listings[all_features + [TARGET]].dropna(subset=[TARGET])
    X = df[all_features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "r2":  round(r2_score(y_test, y_pred), 3),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
    }
    print(f"LightGBM  R²={metrics['r2']}  MAE=€{metrics['mae']}")
    return pipe, metrics
@lru_cache(maxsize=1)
def get_feature_importances(listings_hash: int) -> pd.DataFrame:
    """Top features from the fitted LightGBM model."""
    pipe, _ = get_model(listings_hash)
    preprocessor = pipe.named_steps["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe = cat_transformer.named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names
    importances = pipe.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": all_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def get_test_predictions(listings_hash: int):
    """Returns (y_test, y_pred) arrays for model evaluation charts."""
    pipe, _ = get_model(listings_hash)
    from data_loader import get_data
    listings = get_data()
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = listings[all_features + [TARGET]].dropna(subset=[TARGET])
    X, y = df[all_features], df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    y_pred = pipe.predict(X_test)
    return y_test.values, y_pred


def predict_price(
    pipeline,
    neighbourhood: str,
    room_type: str,
    property_type: str,
    accommodates: int,
    min_nights: int,
    listings: pd.DataFrame,
) -> float | None:
    medians = listings[NUMERIC_FEATURES].median()
    row = {f: medians[f] for f in NUMERIC_FEATURES}
    row["accommodates"]          = accommodates
    row["minimum_nights_capped"] = min(min_nights, 30)
    row["neighbourhood_cleansed"] = neighbourhood
    row["room_type"]              = room_type
    row["property_type_grouped"]  = property_type
    row["host_is_superhost"]      = True
    row["instant_bookable"]       = False
    X = pd.DataFrame([row])
    try:
        pred = pipeline.predict(X)[0]
        return round(float(max(pred, 10)), 2)
    except Exception:
        return None
