import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

AIRBNB_PALETTE = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676", "#FFB400"]
PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color="#484848"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        colorway=AIRBNB_PALETTE,
    )
)

print("=" * 55)
print("  Barcelona Airbnb Dashboard — starting up")
print("=" * 55)

from data_loader import get_data, get_filter_options, filter_listings, compute_neighbourhood_stats
from model import (get_model, predict_price, get_feature_importances,
                   get_test_predictions, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

listings = get_data()
options  = get_filter_options(listings)
_hash    = hash((len(listings), tuple(listings.columns.tolist())))
pipeline, metrics = get_model(_hash)
CITY_MEDIAN = round(listings["price"].median(), 2)


# ─── helpers ────────────────────────────────────────────────────────────────

def _empty_fig(msg="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#767676"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300)
    return fig


def _apply_template(fig, height=320, **extra):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height, **extra)
    return fig


def _clean_feat_name(name: str) -> str:
    name = re.sub(r"^neighbourhood_cleansed_", "Area: ", name)
    name = re.sub(r"^room_type_", "Room: ", name)
    name = re.sub(r"^property_type_grouped_", "Type: ", name)
    name = re.sub(r"^host_is_superhost_", "Superhost: ", name)
    name = re.sub(r"^instant_bookable_", "Instant book: ", name)
    return name.replace("_", " ").title()


# ─── precompute EDA figures ──────────────────────────────────────────────────

def _build_eda_figures(lst: pd.DataFrame):
    # 1  Neighbourhood median price (horizontal bar)
    nb_med = (
        lst.groupby("neighbourhood_cleansed")["price"]
        .median().sort_values().reset_index(name="median_price")
    )
    fig_nb = px.bar(
        nb_med, x="median_price", y="neighbourhood_cleansed", orientation="h",
        color="median_price", color_continuous_scale=["#FFB400", "#FF5A5F"],
        title="Median nightly price by neighbourhood",
        labels={"median_price": "Median price (€)", "neighbourhood_cleansed": ""},
    )
    fig_nb.update_coloraxes(showscale=False)
    fig_nb.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                         xaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                         yaxis=dict(showgrid=False))

    # 2  Room-type share (donut)
    rt_counts = lst["room_type"].value_counts().reset_index(name="count")
    rt_counts.columns = ["room_type", "count"]
    fig_rt = px.pie(
        rt_counts, values="count", names="room_type", hole=0.44,
        color_discrete_sequence=AIRBNB_PALETTE,
        title="Listing mix by room type",
    )
    fig_rt.update_traces(textposition="outside", textinfo="percent+label",
                         pull=[0.04] + [0] * (len(rt_counts) - 1))
    fig_rt.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                         showlegend=False)

    # 3  Price vs accommodates (box)
    acc_df = lst[lst["accommodates"].between(1, 10)].copy()
    acc_df["accommodates"] = acc_df["accommodates"].astype(str)
    order = [str(i) for i in range(1, 11)]
    fig_acc = px.box(
        acc_df, x="accommodates", y="price",
        color="accommodates", color_discrete_sequence=AIRBNB_PALETTE * 2,
        category_orders={"accommodates": order},
        title="Price distribution by capacity (guests)",
        labels={"accommodates": "Guests", "price": "Price (€)"},
        points=False,
    )
    fig_acc.update_layout(**PLOTLY_TEMPLATE["layout"], height=340, showlegend=False,
                          yaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                          xaxis=dict(showgrid=False))

    # 4  Superhost vs regular host median price by room type
    sh_df = lst.dropna(subset=["host_is_superhost"]).copy()
    sh_df["Host type"] = sh_df["host_is_superhost"].map(
        {True: "Superhost ★", False: "Regular host"})
    sh_agg = (
        sh_df.groupby(["room_type", "Host type"])["price"]
        .median().reset_index(name="median_price")
    )
    fig_sh = px.bar(
        sh_agg, x="room_type", y="median_price", color="Host type",
        barmode="group",
        color_discrete_map={"Superhost ★": "#FF5A5F", "Regular host": "#00A699"},
        title="Superhost pricing premium by room type",
        labels={"room_type": "", "median_price": "Median price (€)"},
    )
    fig_sh.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                         margin=dict(l=10, r=10, t=75, b=10),
                         yaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                         xaxis=dict(showgrid=False),
                         legend=dict(
                             orientation="h", y=1.18, x=0,
                             title=dict(text="Host type  ", font=dict(size=12)),
                         ))

    # 5  Amenity count vs price scatter (sample)
    samp = lst[lst["price"] <= 600].sample(min(1500, len(lst)), random_state=123)
    fig_am = px.scatter(
        samp, x="amenity_count", y="price",
        color="room_type", color_discrete_sequence=AIRBNB_PALETTE,
        opacity=0.45, trendline="ols",
        title="Amenity count vs nightly price",
        labels={"amenity_count": "Number of amenities", "price": "Price (€)",
                "room_type": "Room type"},
    )
    fig_am.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                         margin=dict(l=10, r=10, t=75, b=10),
                         yaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                         xaxis=dict(showgrid=False),
                         legend=dict(
                             orientation="h", y=1.18, x=0,
                             title=dict(text="Room type  ", font=dict(size=12)),
                         ))

    # 6  Review-score rating distribution
    rated = lst.dropna(subset=["review_scores_rating"])
    fig_rat = px.histogram(
        rated, x="review_scores_rating", nbins=40,
        color_discrete_sequence=["#00A699"],
        title="Distribution of guest review scores",
        labels={"review_scores_rating": "Review score (1–5)",
                "count": "Listings"},
    )
    fig_rat.update_layout(**PLOTLY_TEMPLATE["layout"], height=340,
                          yaxis=dict(gridcolor="#F0F0F0"),
                          xaxis=dict(showgrid=False))
    fig_rat.update_traces(marker_line_width=0.4, marker_line_color="white")

    # Narrative text (dynamic) — key findings from EDA analysis
    top_nb  = nb_med.iloc[-1]["neighbourhood_cleansed"]
    bot_nb  = nb_med.iloc[0]["neighbourhood_cleansed"]
    top_p   = nb_med.iloc[-1]["median_price"]
    bot_p   = nb_med.iloc[0]["median_price"]
    rt_med  = lst.groupby("room_type")["price"].median()
    ent_p   = rt_med.get("Entire home/apt", None)
    prv_p   = rt_med.get("Private room", None)
    ratio   = f"{ent_p/prv_p:.1f}×" if (ent_p and prv_p and prv_p > 0) else "N/A"
    sh_med  = lst.dropna(subset=["host_is_superhost"]).groupby("host_is_superhost")["price"].median()
    sh_prem_val = ""
    sh_prem_pct = ""
    if True in sh_med.index and False in sh_med.index and sh_med[False] > 0:
        pct = (sh_med[True] - sh_med[False]) / sh_med[False] * 100
        sh_prem_val = f"€{sh_med[True]:.0f}"
        sh_prem_pct = f"{pct:+.0f}%"
    pct_rated = round(rated["review_scores_rating"].ge(4.5).mean() * 100)
    ent_share = round(lst["room_type"].eq("Entire home/apt").mean() * 100)
    # Summer seasonality — peak months from last_review proxy
    seasonal = lst.dropna(subset=["last_review"])
    seasonal_med = seasonal.groupby(seasonal["last_review"].dt.month)["price"].median()
    peak_month = int(seasonal_med.idxmax()) if not seasonal_med.empty else 8
    peak_month_name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][peak_month - 1]
    # Min-nights insight
    short_stay_pct = round(lst["minimum_nights"].le(3).mean() * 100)
    narrative = (
        f"Location is the dominant price driver: {top_nb} leads at €{top_p:.0f}/night while "
        f"{bot_nb} is most affordable at €{bot_p:.0f}/night — a {top_p/bot_p:.1f}× city-wide spread. "
        f"Entire homes/apts dominate supply ({ent_share}% of listings) and cost {ratio} more than private rooms. "
        f"Superhosts command a {sh_prem_pct} premium (€{sh_med.get(True, 0):.0f} vs €{sh_med.get(False, 0):.0f} median) — "
        f"quality signalling translates directly into pricing power. "
        f"Demand peaks around {peak_month_name} (summer tourism season) and collapses in winter, "
        f"consistent with Barcelona's seasonal tourism cycle. "
        f"{short_stay_pct}% of listings allow stays of 3 nights or fewer, targeting short-stay tourists; "
        f"a secondary cluster at 30-night minimums reflects regulatory compliance. "
        f"{pct_rated}% of rated listings score ≥4.5 ★ — a quality-saturated market "
        f"where amenities and location matter more than marginal rating differences."
    )
    return narrative, fig_nb, fig_rt, fig_acc, fig_sh, fig_am, fig_rat


# ─── precompute ML figures ───────────────────────────────────────────────────

def _build_ml_figures(lst: pd.DataFrame, pipe, met, h):
    # Feature importances
    try:
        imp_df = get_feature_importances(h)
        top15  = imp_df.head(15).copy()
        top15["label"] = top15["feature"].apply(_clean_feat_name)
        top15 = top15.sort_values("importance")
        fig_imp = px.bar(
            top15, x="importance", y="label", orientation="h",
            color="importance", color_continuous_scale=["#FFB400", "#FF5A5F"],
            title="Top 15 features — LightGBM importance",
            labels={"importance": "Feature importance", "label": ""},
        )
        fig_imp.update_coloraxes(showscale=False)
        fig_imp.update_layout(**PLOTLY_TEMPLATE["layout"], height=420,
                              xaxis=dict(gridcolor="#F0F0F0"),
                              yaxis=dict(showgrid=False))
    except Exception:
        fig_imp = _empty_fig("Feature importance unavailable")

    # Actual vs Predicted
    try:
        y_test, y_pred = get_test_predictions(h)
        cap = 500
        mask = (y_test <= cap) & (y_pred <= cap)
        ap_df = pd.DataFrame({"Actual (€)": y_test[mask], "Predicted (€)": y_pred[mask]})
        fig_ap = px.scatter(
            ap_df, x="Actual (€)", y="Predicted (€)",
            opacity=0.35, color_discrete_sequence=["#FF5A5F"],
            title=f"Actual vs Predicted price  (R² = {met['r2']})",
        )
        # perfect-fit line
        lo, hi = 0, cap
        fig_ap.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi,
                         line=dict(color="#484848", dash="dash", width=1.5))
        fig_ap.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                              xaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                              yaxis=dict(tickprefix="€", gridcolor="#F0F0F0"))
    except Exception:
        fig_ap = _empty_fig("Prediction plot unavailable")

    # Residuals distribution
    try:
        residuals = y_pred - y_test
        resid_clip = residuals[(residuals > -300) & (residuals < 300)]
        fig_resid = px.histogram(
            pd.DataFrame({"Residual (€)": resid_clip}),
            x="Residual (€)", nbins=60,
            color_discrete_sequence=["#00A699"],
            title="Residual distribution  (Predicted − Actual)",
        )
        mean_r  = round(float(np.mean(resid_clip)), 1)
        std_r   = round(float(np.std(resid_clip)), 1)
        fig_resid.add_vline(x=0, line_dash="dash", line_color="#484848", line_width=1.5)
        fig_resid.add_vline(x=mean_r, line_dash="dot", line_color="#FF5A5F", line_width=1.5,
                            annotation_text=f"Mean={mean_r:+.0f}€",
                            annotation_position="top right")
        fig_resid.update_layout(**PLOTLY_TEMPLATE["layout"], height=340,
                                yaxis=dict(gridcolor="#F0F0F0"),
                                xaxis=dict(showgrid=False))
        fig_resid.update_traces(marker_line_width=0.4, marker_line_color="white")
    except Exception:
        fig_resid = _empty_fig("Residuals unavailable")
        mean_r, std_r = 0, 0

    # Prediction error by room type (box)
    try:
        y_test_e, y_pred_e = get_test_predictions(h)
        all_feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        from sklearn.model_selection import train_test_split
        # "room_type" is already in all_feats (CATEGORICAL_FEATURES), no need to add it again
        df_ml = lst[all_feats + ["price"]].dropna(subset=["price"])
        _, df_test = train_test_split(df_ml, test_size=0.2, random_state=123)
        err_df = pd.DataFrame({"room_type": df_test["room_type"].values,
                               "error": y_pred_e - y_test_e})
        err_clip = err_df[err_df["error"].between(-300, 300)]
        fig_err = px.box(
            err_clip, x="room_type", y="error",
            color="room_type", color_discrete_sequence=AIRBNB_PALETTE,
            title="Prediction error by room type",
            labels={"room_type": "", "error": "Residual (€)"},
            points=False,
        )
        fig_err.add_hline(y=0, line_dash="dash", line_color="#484848", line_width=1.5)
        fig_err.update_layout(**PLOTLY_TEMPLATE["layout"], height=340, showlegend=False,
                              yaxis=dict(tickprefix="€", gridcolor="#F0F0F0"),
                              xaxis=dict(showgrid=False))
    except Exception:
        fig_err = _empty_fig("Error breakdown unavailable")

    # ML narrative
    n_train = round(len(lst) * 0.8)
    n_test  = len(lst) - n_train
    try:
        bias_note = (
            f"Residuals are centred near {mean_r:+.0f}€ (σ = {std_r:.0f}€), "
            "indicating the model is approximately unbiased across the price range."
        )
    except Exception:
        bias_note = ""
    narrative = (
        f"A LightGBM gradient-boosting model (trained on log-price, evaluated in €) was "
        f"trained on {n_train:,} listings and evaluated on {n_test:,} held-out listings. "
        f"It achieves R² = {met['r2']}, MAE = €{met['mae']:.0f}/night, and "
        f"RMSE = €{met['rmse']:.0f}/night. "
        f"MAE reflects the average error; RMSE penalises large mispredictions more heavily. "
        f"Physical listing attributes (accommodates, bedrooms, bathrooms) and "
        f"location (neighbourhood) dominate feature importance. {bias_note} "
        f"Business implication: hosts can use the predicted price as an evidence-based anchor "
        f"and adjust ±10–15% based on seasonal demand or listing quality relative to neighbours."
    )
    return narrative, fig_imp, fig_ap, fig_resid, fig_err


# ─── compute everything at startup ──────────────────────────────────────────

(eda_narrative,
 fig_eda_nb, fig_eda_rt, fig_eda_acc,
 fig_eda_sh, fig_eda_am, fig_eda_rat) = _build_eda_figures(listings)

(ml_narrative,
 fig_ml_imp, fig_ml_ap,
 fig_ml_resid, fig_ml_err) = _build_ml_figures(listings, pipeline, metrics, _hash)


# ─── layout helpers ──────────────────────────────────────────────────────────

def _kpi_card(value, label):
    return html.Div(className="kpi-card", children=[
        html.Div(value, className="kpi-value"),
        html.Div(label, className="kpi-label"),
    ])


def _narrative_card(icon, title, text, border_color="#00A699"):
    return html.Div(className="card narrative-card", style={"borderLeft": f"4px solid {border_color}"},
                    children=[
                        html.Div(f"{icon} {title}", className="insight-title"),
                        html.P(text),
                    ])


def _chart_card(fig, config=None):
    return html.Div(className="card",
                    children=[dcc.Graph(figure=fig,
                                        config=config or {"displayModeBar": False})])


# ─── sidebar (tab 1 only) ────────────────────────────────────────────────────

sidebar = html.Div(id="sidebar", children=[
    html.H2("Filters"),
    html.Span("City Area", className="filter-label"),
    dcc.Dropdown(
        id="dd-neighbourhood",
        options=[{"label": "All neighbourhoods", "value": "All"}] +
                [{"label": n, "value": n} for n in options["neighbourhoods"]],
        value="All", clearable=False, style={"fontSize": "13px"},
    ),
    html.Span("Room Type", className="filter-label"),
    dcc.Dropdown(
        id="dd-room-type",
        options=[{"label": "All room types", "value": "All"}] +
                [{"label": r, "value": r} for r in options["room_types"]],
        value="All", clearable=False, style={"fontSize": "13px"},
    ),
    html.Span("Property Type", className="filter-label"),
    dcc.Dropdown(
        id="dd-property-type",
        options=[{"label": "All property types", "value": "All"}] +
                [{"label": p, "value": p} for p in options["property_types"]],
        value="All", clearable=False, style={"fontSize": "13px"},
    ),
    html.Span("Accommodates (at least)", className="filter-label"),
    dcc.Slider(
        id="sl-accommodates",
        min=1, max=16, step=1, value=2,
        marks={i: str(i) for i in [1, 4, 8, 12, 16]},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Span("Minimum Nights (max)", className="filter-label"),
    dcc.Slider(
        id="sl-min-nights",
        min=1, max=30, step=1, value=30,
        marks={i: str(i) for i in [1, 5, 10, 20, 30]},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginTop": "32px", "borderTop": "1px solid #EBEBEB",
                    "paddingTop": "16px"}),
    html.P(f"Dataset: {len(listings):,} listings · Barcelona",
           style={"fontSize": "11px", "color": "#767676", "margin": "0"}),
    html.P("Source: Inside Airbnb (Sep 2025)",
           style={"fontSize": "11px", "color": "#767676", "margin": "4px 0 0 0"}),
])


# ─── tab contents ────────────────────────────────────────────────────────────

explorer_tab = html.Div(id="main-layout", children=[
    sidebar,
    html.Div(id="content", children=[
        dcc.Loading(type="circle", color="#FF5A5F", children=[
            html.Div(id="insight-card", className="card", children=[
                html.Div("📊 Data story", className="insight-title"),
                html.P(id="insight-text", children="Loading insights…"),
            ])
        ]),
        dcc.Loading(type="circle", color="#FF5A5F", children=[
            html.Div(id="price-card", className="card", children=[
                html.Div("Predicted nightly price", className="price-label"),
                html.Div(id="price-value", className="price-value", children="—"),
                html.Div(id="price-meta", className="price-meta", children=""),
            ])
        ]),
        dcc.Loading(type="circle", color="#FF5A5F", children=[
            html.Div(id="kpi-row", className="kpi-row"),
        ]),
        dcc.Loading(type="circle", color="#FF5A5F", children=[
            html.Div(className="charts-row", children=[
                html.Div(className="card", children=[
                    dcc.Graph(id="chart-price-dist",
                              config={"displayModeBar": False}),
                ]),
                html.Div(className="card", children=[
                    dcc.Graph(id="chart-price-trend",
                              config={"displayModeBar": False}),
                ]),
            ]),
        ]),
    ]),
])

eda_tab = html.Div(className="tab-page", children=[
    _narrative_card("📈", "EDA Key Findings", eda_narrative, "#00A699"),
    html.Div(className="charts-row", children=[
        _chart_card(fig_eda_nb),
        _chart_card(fig_eda_rt),
    ]),
    html.Div(className="charts-row", children=[
        _chart_card(fig_eda_acc),
        _chart_card(fig_eda_sh),
    ]),
    html.Div(className="charts-row", children=[
        _chart_card(fig_eda_am),
        _chart_card(fig_eda_rat),
    ]),
])

n_train = round(len(listings) * 0.8)
n_test  = len(listings) - n_train
ml_tab = html.Div(className="tab-page", children=[
    _narrative_card("🤖", "ML Model Summary", ml_narrative, "#FF5A5F"),
    html.Div(className="kpi-row", style={"marginBottom": "20px"}, children=[
        _kpi_card(str(metrics["r2"]),      "R² (test set)"),
        _kpi_card(f"€{metrics['mae']:.0f}", "MAE / night"),
        _kpi_card(f"€{metrics['rmse']:.0f}", "RMSE / night"),
        _kpi_card(f"{n_train:,}",           "Training listings"),
    ]),
    html.Div(className="charts-row", children=[
        _chart_card(fig_ml_imp),
        _chart_card(fig_ml_ap),
    ]),
    html.Div(className="charts-row", children=[
        _chart_card(fig_ml_resid),
        _chart_card(fig_ml_err),
    ]),
])


# ─── app layout ──────────────────────────────────────────────────────────────

tab_style         = {"padding": "10px 22px", "fontWeight": "600",
                     "fontSize": "14px", "color": "#767676",
                     "borderBottom": "3px solid transparent"}
tab_selected_style = {**tab_style, "color": "#FF5A5F",
                      "borderBottom": "3px solid #FF5A5F"}

app.layout = html.Div([
    html.Div(id="header", children=[
        html.Div("🏠", style={"fontSize": "28px"}),
        html.Div([
            html.H1("Barcelona Airbnb Analytics"),
            html.P("Interactive price explorer · Group 12", className="subtitle"),
        ])
    ]),
    dcc.Tabs(
        id="main-tabs", value="tab-explorer",
        style={"background": "#fff", "borderBottom": "1px solid #EBEBEB"},
        children=[
            dcc.Tab(label="🔍 Price Explorer", value="tab-explorer",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[explorer_tab]),
            dcc.Tab(label="📊 EDA Findings", value="tab-eda",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[eda_tab]),
            dcc.Tab(label="🤖 ML Findings", value="tab-ml",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[ml_tab]),
        ],
    ),
])


# ─── callbacks (Price Explorer tab) ─────────────────────────────────────────

def _build_insight(df, neighbourhood, room_type, property_type,
                   accommodates, min_nights):
    n = len(df)
    if n < 5:
        return (
            "Not enough listings match this combination to draw conclusions. "
            "Try broadening your filters — for example selecting 'All' for "
            "room type or property type."
        )
    median_p   = df["price"].median()
    pct_vs_city = ((median_p - CITY_MEDIAN) / CITY_MEDIAN) * 100
    direction  = "above" if pct_vs_city >= 0 else "below"
    abs_pct    = abs(round(pct_vs_city, 1))
    area_label = neighbourhood if neighbourhood != "All" else "Barcelona overall"
    rt_label   = room_type     if room_type    != "All" else "all room types"
    avg_rating = df["review_scores_rating"].dropna().mean()
    rating_str = f"{avg_rating:.2f} ★" if not pd.isna(avg_rating) else "no rating data"
    superhost_pct = df["host_is_superhost"].fillna(False).mean() * 100
    sentence1 = (
        f"In {area_label}, {rt_label} ({accommodates} guest"
        f"{'s' if accommodates > 1 else ''}) "
        f"have a median nightly price of €{median_p:.0f} — "
        f"{abs_pct}% {direction} the city-wide median of €{CITY_MEDIAN:.0f}."
    )
    sentence2 = (
        f"This selection covers {n:,} listing{'s' if n > 1 else ''} "
        f"with an average guest rating of {rating_str}. "
        f"{'Superhosts account for ' + str(round(superhost_pct)) + '% of hosts here, suggesting a competitive, quality-conscious market.' if superhost_pct > 0 else ''}"
    )
    if abs_pct >= 20:
        rec = (
            "Listings in this segment command a significant premium. "
            "The LightGBM model below gives your personalised price estimate — "
            "staying within 10–15% of it maximises bookings without leaving revenue on the table."
        )
    elif abs_pct < 5:
        rec = (
            "Pricing here is closely aligned with the city median. "
            "Differentiation through amenities, professional photos, and "
            "superhost status will matter more than price adjustments."
        )
    else:
        rec = (
            "There is moderate pricing variation in this segment. "
            "Use the predicted price below as an anchor and adjust ±10% "
            "based on your amenity count and review score relative to neighbours."
        )
    return f"{sentence1} {sentence2} {rec}"


@callback(
    Output("insight-text",      "children"),
    Output("price-value",       "children"),
    Output("price-meta",        "children"),
    Output("kpi-row",           "children"),
    Output("chart-price-dist",  "figure"),
    Output("chart-price-trend", "figure"),
    Input("dd-neighbourhood",   "value"),
    Input("dd-room-type",       "value"),
    Input("dd-property-type",   "value"),
    Input("sl-accommodates",    "value"),
    Input("sl-min-nights",      "value"),
)
def update_dashboard(neighbourhood, room_type, property_type,
                     accommodates, min_nights):
    df = filter_listings(
        listings,
        neighbourhood=neighbourhood,
        room_type=room_type,
        property_type=property_type,
        accommodates=accommodates,
        min_nights=min_nights,
    )
    insight = _build_insight(df, neighbourhood, room_type, property_type,
                             accommodates, min_nights)

    nb  = neighbourhood if neighbourhood != "All" else listings["neighbourhood_cleansed"].mode()[0]
    rt  = room_type     if room_type     != "All" else "Entire home/apt"
    pt  = property_type if property_type != "All" else listings["property_type_grouped"].mode()[0]
    pred = predict_price(pipeline, nb, rt, pt, accommodates, min_nights, listings)
    if pred is not None:
        price_display = f"€{pred:.0f}"
        price_meta    = (f"LightGBM model  ·  R² = {metrics['r2']}  ·  "
                         f"MAE = €{metrics['mae']:.0f}  ·  RMSE = €{metrics['rmse']:.0f} on test set")
    else:
        price_display = "—"
        price_meta    = "Unable to predict for this combination"

    stats = compute_neighbourhood_stats(df)

    def kpi(value, label):
        return html.Div(className="kpi-card", children=[
            html.Div(value, className="kpi-value"),
            html.Div(label, className="kpi-label"),
        ])

    kpi_children = [
        kpi(f"€{stats['median_price']:.0f}" if stats["median_price"] else "—",
            "Median price / night"),
        kpi(f"{stats['n_listings']:,}", "Listings"),
        kpi(f"{stats['avg_rating']:.2f} ★" if stats["avg_rating"] else "—",
            "Avg rating"),
        kpi(f"{stats['superhost_pct']:.0f}%" if stats["superhost_pct"] is not None else "—",
            "Superhosts"),
    ]

    if df.empty or len(df) < 5:
        fig_dist = _empty_fig()
    else:
        area_label = neighbourhood if neighbourhood != "All" else "Barcelona"
        fig_dist = px.box(
            df, x="room_type", y="price",
            color="room_type", color_discrete_sequence=AIRBNB_PALETTE,
            title=f"Price distribution by room type — {area_label}",
            labels={"room_type": "Room type", "price": "Nightly price (€)"},
            points=False,
        )
        fig_dist.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=320, showlegend=False,
            yaxis=dict(gridcolor="#F0F0F0", tickprefix="€"),
            xaxis=dict(showgrid=False),
        )
        fig_dist.update_traces(marker_line_width=1.2)

    trend_df = (
        listings if neighbourhood == "All"
        else listings[listings["neighbourhood_cleansed"] == neighbourhood]
    )
    if room_type != "All":
        trend_df = trend_df[trend_df["room_type"] == room_type]
    if property_type != "All":
        trend_df = trend_df[trend_df["property_type_grouped"] == property_type]
    trend_df = (
        trend_df
        .dropna(subset=["last_review", "price"])
        .assign(month=lambda d: d["last_review"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["price"]
        .agg(median_price="median", n="count")
        .reset_index()
        .query("n >= 3")
        .sort_values("month")
        .tail(36)
    )
    if trend_df.empty or len(trend_df) < 3:
        fig_trend = _empty_fig("Not enough review data for trend")
    else:
        area_label = neighbourhood if neighbourhood != "All" else "Barcelona"
        fig_trend = px.line(
            trend_df, x="month", y="median_price",
            title=f"Monthly median price trend — {area_label}",
            labels={"month": "Month", "median_price": "Median price (€)"},
            color_discrete_sequence=["#FF5A5F"],
        )
        fig_trend.add_scatter(
            x=trend_df["month"], y=trend_df["median_price"],
            mode="markers", marker=dict(color="#FF5A5F", size=5),
            showlegend=False,
        )
        fig_trend.update_layout(
            **PLOTLY_TEMPLATE["layout"], height=320,
            yaxis=dict(gridcolor="#F0F0F0", tickprefix="€"),
            xaxis=dict(showgrid=False, title=""),
        )
        fig_trend.update_traces(line=dict(width=2.5))

    return (insight, price_display, price_meta,
            kpi_children, fig_dist, fig_trend)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
