import os
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
        margin=dict(l=10, r=10, t=36, b=10),
        colorway=AIRBNB_PALETTE,
    )
)
print("=" * 55)
print("  Barcelona Airbnb Dashboard — starting up")
print("=" * 55)
from data_loader import get_data, get_filter_options, filter_listings, compute_neighbourhood_stats
from model import get_model, predict_price, NUMERIC_FEATURES, CATEGORICAL_FEATURES
listings = get_data()
options  = get_filter_options(listings)
_hash = hash((len(listings), tuple(listings.columns.tolist())))
pipeline, metrics = get_model(_hash)
CITY_MEDIAN = round(listings["price"].median(), 2)
sidebar = html.Div(id="sidebar", children=[
    html.H2("Filters"),
    html.Span("City Area", className="filter-label"),
    dcc.Dropdown(
        id="dd-neighbourhood",
        options=[{"label": "All neighbourhoods", "value": "All"}] +
                [{"label": n, "value": n} for n in options["neighbourhoods"]],
        value="All",
        clearable=False,
        style={"fontSize": "13px"},
    ),
    html.Span("Room Type", className="filter-label"),
    dcc.Dropdown(
        id="dd-room-type",
        options=[{"label": "All room types", "value": "All"}] +
                [{"label": r, "value": r} for r in options["room_types"]],
        value="All",
        clearable=False,
        style={"fontSize": "13px"},
    ),
    html.Span("Property Type", className="filter-label"),
    dcc.Dropdown(
        id="dd-property-type",
        options=[{"label": "All property types", "value": "All"}] +
                [{"label": p, "value": p} for p in options["property_types"]],
        value="All",
        clearable=False,
        style={"fontSize": "13px"},
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
        min=1, max=30, step=1, value=5,
        marks={i: str(i) for i in [1, 5, 10, 20, 30]},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginTop": "32px", "borderTop": "1px solid #EBEBEB",
                    "paddingTop": "16px"}),
    html.P(
        f"Dataset: {len(listings):,} listings · Barcelona",
        style={"fontSize": "11px", "color": "#767676", "margin": "0"},
    ),
    html.P(
        "Source: Inside Airbnb (Sep 2025)",
        style={"fontSize": "11px", "color": "#767676", "margin": "4px 0 0 0"},
    ),
])
app.layout = html.Div([
    html.Div(id="header", children=[
        html.Div("🏠", style={"fontSize": "28px"}),
        html.Div([
            html.H1("Barcelona Airbnb Analytics"),
            html.P("Interactive price explorer · Group 12", className="subtitle"),
        ])
    ]),
    html.Div(id="main-layout", children=[
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
    ]),
])
def _empty_fig(msg="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#767676"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300)
    return fig
def _build_insight(df, neighbourhood, room_type, property_type,
                   accommodates, min_nights):
    n = len(df)
    if n < 5:
        return (
            "Not enough listings match this combination to draw conclusions. "
            "Try broadening your filters — for example selecting 'All' for "
            "room type or property type."
        )
    median_p = df["price"].median()
    pct_vs_city = ((median_p - CITY_MEDIAN) / CITY_MEDIAN) * 100
    direction = "above" if pct_vs_city >= 0 else "below"
    abs_pct = abs(round(pct_vs_city, 1))
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
                         f"MAE = €{metrics['mae']:.0f} on test set")
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
            df,
            x="room_type",
            y="price",
            color="room_type",
            color_discrete_sequence=AIRBNB_PALETTE,
            title=f"Price distribution by room type — {area_label}",
            labels={"room_type": "Room type", "price": "Nightly price (€)"},
            points=False,
        )
        fig_dist.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=320,
            showlegend=False,
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
            trend_df,
            x="month",
            y="median_price",
            title=f"Monthly median price trend — {area_label}",
            labels={"month": "Month", "median_price": "Median price (€)"},
            color_discrete_sequence=["#FF5A5F"],
        )
        fig_trend.add_scatter(
            x=trend_df["month"],
            y=trend_df["median_price"],
            mode="markers",
            marker=dict(color="#FF5A5F", size=5),
            showlegend=False,
        )
        fig_trend.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            height=320,
            yaxis=dict(gridcolor="#F0F0F0", tickprefix="€"),
            xaxis=dict(showgrid=False, title=""),
        )
        fig_trend.update_traces(line=dict(width=2.5))
    return (insight, price_display, price_meta,
            kpi_children, fig_dist, fig_trend)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
