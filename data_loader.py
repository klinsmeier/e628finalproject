import gzip, io, re, warnings
import numpy as np
import pandas as pd
import requests
from functools import lru_cache
warnings.filterwarnings("ignore")
LISTINGS_URL = "https://data.insideairbnb.com/spain/catalonia/barcelona/2025-09-14/data/listings.csv.gz"
REVIEWS_URL  = "https://data.insideairbnb.com/spain/catalonia/barcelona/2025-09-14/data/reviews.csv.gz"
SEED = 123
AIRBNB_PALETTE = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676", "#FFB400"]
def load_gz_csv(url: str) -> pd.DataFrame:
    print(f"  Downloading {url.split('/')[-1]} ...", end=" ")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(gzip.open(io.BytesIO(r.content)), low_memory=False)
    print(f"{len(df):,} rows x {df.shape[1]} cols")
    return df

def _generate_synthetic_listings(n: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic Barcelona Airbnb data."""
    rng = np.random.default_rng(SEED)
    neighbourhoods = [
        "Eixample", "Gràcia", "Sant Martí", "Sants-Montjuïc", "Sarrià-Sant Gervasi",
        "Nou Barris", "Sant Andreu", "Horta-Guinardó", "Les Corts", "Ciutat Vella",
    ]
    room_types = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    room_type_weights = [0.60, 0.30, 0.05, 0.05]
    property_types = [
        "Entire rental unit", "Private room in rental unit", "Entire condo",
        "Entire loft", "Private room in home", "Entire home", "Shared room",
        "Entire serviced apartment", "Room in hotel", "Entire guesthouse", "Other"
    ]
    host_ids = rng.integers(1000, 99999, size=n)
    room_type_arr = rng.choice(room_types, size=n, p=room_type_weights)
    nb_arr = rng.choice(neighbourhoods, size=n)
    pt_arr = rng.choice(property_types, size=n, p=[0.30,0.18,0.10,0.08,0.10,0.07,0.03,0.05,0.03,0.03,0.03])
    base_price = np.where(room_type_arr == "Entire home/apt", 100,
                 np.where(room_type_arr == "Private room", 45,
                 np.where(room_type_arr == "Hotel room", 80, 25)))
    price = np.clip(rng.lognormal(np.log(base_price), 0.5, n), 10, 1000).round(2)
    accommodates = np.where(room_type_arr == "Entire home/apt",
                            rng.integers(2, 9, n), rng.integers(1, 4, n))
    bedrooms = np.clip(rng.integers(1, 5, n), 1, accommodates)
    beds = np.clip(rng.integers(1, 7, n), bedrooms, accommodates + 1)
    bathrooms_text = [f"{rng.integers(1,3)}.0 bath{'s' if rng.integers(0,2) else ''}" for _ in range(n)]
    amenity_count = rng.integers(5, 60, n)
    amenities = ['"' + '","'.join([f"amenity_{j}" for j in range(c)]) + '"' for c in amenity_count]
    min_nights = rng.choice([1,2,3,5,7,14,30], size=n, p=[0.35,0.20,0.15,0.10,0.10,0.05,0.05])
    avail_365 = rng.integers(0, 366, n)
    reviews_pm = np.clip(rng.exponential(1.0, n), 0, 10).round(2)
    n_reviews = rng.integers(0, 200, n)
    rating = np.where(n_reviews > 0, np.clip(rng.normal(4.6, 0.3, n), 1, 5).round(2), np.nan)
    clean = np.where(n_reviews > 0, np.clip(rng.normal(4.6, 0.3, n), 1, 5).round(2), np.nan)
    loc   = np.where(n_reviews > 0, np.clip(rng.normal(4.7, 0.25, n), 1, 5).round(2), np.nan)
    host_since_days = rng.integers(365, 365*12, n)
    host_since = pd.to_datetime("2025-09-14") - pd.to_timedelta(host_since_days, unit="D")
    last_review_days = rng.integers(0, 730, n)
    last_review = pd.to_datetime("2025-09-14") - pd.to_timedelta(last_review_days, unit="D")
    last_review = pd.Series(last_review).where(pd.Series(n_reviews) > 0, other=pd.NaT)
    superhost = rng.choice(["t", "f"], size=n, p=[0.35, 0.65])
    instant = rng.choice(["t", "f"], size=n, p=[0.55, 0.45])
    host_listings = rng.integers(1, 20, n)
    resp_rate = np.clip(rng.normal(92, 10, n), 0, 100).round(0).astype(int).astype(str) + "%"
    acc_rate  = np.clip(rng.normal(85, 15, n), 0, 100).round(0).astype(int).astype(str) + "%"
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "host_id": host_ids,
        "host_since": host_since.strftime("%Y-%m-%d"),
        "host_is_superhost": superhost,
        "host_response_rate": resp_rate,
        "host_acceptance_rate": acc_rate,
        "host_listings_count": host_listings,
        "calculated_host_listings_count": host_listings,
        "neighbourhood_cleansed": nb_arr,
        "latitude":  rng.uniform(41.33, 41.47, n),
        "longitude": rng.uniform(2.10, 2.23, n),
        "property_type": pt_arr,
        "room_type": room_type_arr,
        "accommodates": accommodates,
        "bathrooms_text": bathrooms_text,
        "bedrooms": bedrooms.astype(float),
        "beds": beds.astype(float),
        "amenities": amenities,
        "price": ["$" + str(p) for p in price],
        "minimum_nights": min_nights,
        "maximum_nights": rng.integers(30, 1125, n),
        "availability_365": avail_365,
        "number_of_reviews": n_reviews,
        "last_review": pd.Series(last_review).dt.strftime("%Y-%m-%d"),
        "reviews_per_month": reviews_pm,
        "review_scores_rating": rating,
        "review_scores_cleanliness": clean,
        "review_scores_location": loc,
        "instant_bookable": instant,
    })
    print(f"{len(df):,} rows x {df.shape[1]} cols (synthetic)")
    return df
def _parse_price(df):
    df = df.copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df
def _filter_price(df, lo=10, hi=1000):
    return df.query("@lo <= price <= @hi")
def _parse_booleans(df):
    for col in ["host_is_superhost", "instant_bookable"]:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False, True: True, False: False})
    return df
def _parse_rates(df):
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
    return df
def _add_features(df):
    df = df.copy()
    df["amenity_count"] = df["amenities"].fillna("").apply(
        lambda x: len(re.findall(r'"([^"]+)"', x))
    )
    df["bathrooms_clean"] = (
        df["bathrooms_text"]
        .fillna("0")
        .str.extract(r"([\d.]+)")[0]
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_tenure_years"] = (
        (pd.Timestamp("2025-09-14") - df["host_since"]).dt.days / 365.25
    )
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["review_month"] = df["last_review"].dt.to_period("M")
    df["beds"] = df["beds"].fillna(1)
    df["bedrooms"] = df["bedrooms"].fillna(1)
    df["minimum_nights_capped"] = df["minimum_nights"].clip(upper=30)
    top_props = df["property_type"].value_counts().head(15).index
    df["property_type_grouped"] = df["property_type"].where(
        df["property_type"].isin(top_props), other="Other"
    )
    return df
def _clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .pipe(_parse_price)
        .pipe(_filter_price)
        .pipe(_parse_booleans)
        .pipe(_parse_rates)
        .pipe(_add_features)
        .query("minimum_nights <= 365")
        .query("accommodates >= 1")
        .reset_index(drop=True)
    )
@lru_cache(maxsize=1)
def get_data():
    print("Loading Barcelona Airbnb data...")
    try:
        raw = load_gz_csv(LISTINGS_URL)
    except Exception as e:
        print(f"  Download failed ({e.__class__.__name__}: {e})")
        print("  Falling back to synthetic demo data...")
        raw = _generate_synthetic_listings(5000)
    listings = _clean_listings(raw)
    print(f"Clean listings: {len(listings):,} rows")
    return listings
def get_filter_options(listings: pd.DataFrame) -> dict:
    neighbourhoods = sorted(listings["neighbourhood_cleansed"].dropna().unique())
    room_types     = sorted(listings["room_type"].dropna().unique())
    prop_types     = sorted(listings["property_type_grouped"].dropna().unique())
    return {
        "neighbourhoods": neighbourhoods,
        "room_types":     room_types,
        "property_types": prop_types,
    }
def filter_listings(
    listings: pd.DataFrame,
    neighbourhood: str | None = None,
    room_type: str | None = None,
    property_type: str | None = None,
    accommodates: int = 2,
    min_nights: int = 1,
) -> pd.DataFrame:
    df = listings.copy()
    if neighbourhood and neighbourhood != "All":
        df = df[df["neighbourhood_cleansed"] == neighbourhood]
    if room_type and room_type != "All":
        df = df[df["room_type"] == room_type]
    if property_type and property_type != "All":
        df = df[df["property_type_grouped"] == property_type]
    df = df[df["accommodates"] == accommodates]
    df = df[df["minimum_nights_capped"] <= min_nights]
    return df
def compute_neighbourhood_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "median_price":   None,
            "n_listings":     0,
            "avg_rating":     None,
            "superhost_pct":  None,
        }
    return {
        "median_price":  round(df["price"].median(), 2),
        "n_listings":    len(df),
        "avg_rating":    round(df["review_scores_rating"].dropna().mean(), 2)
                         if df["review_scores_rating"].notna().any() else None,
        "superhost_pct": round(
            df["host_is_superhost"].fillna(False).mean() * 100, 1
        ),
    }
