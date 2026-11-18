import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Procurement Spend Intelligence",
    page_icon="??",
    layout="wide"
)

st.title("AI-Powered Procurement Spend Intelligence Engine")
st.caption("Multi-company procurement analytics and classification dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    company_a = pd.read_csv(
        "outputs/company_a_procurement_master.csv",
        low_memory=False
    )

    company_b = pd.read_csv(
        "outputs/company_b_procurement_master.csv",
        low_memory=False
    )

    # Normalize Company A
    a = pd.DataFrame({
        "company": "Company A",
        "record_id": company_a["record_id"],
        "supplier": company_a["supplier_name"].fillna("UNKNOWN"),
        "spend": pd.to_numeric(company_a["spend_zar"], errors="coerce"),
        "date": pd.to_datetime(company_a["transaction_date"], errors="coerce", format="%Y-%m-%d"),
        "category_l1": company_a["category_l1"],
        "category_l2": company_a["category_l2"],
        "category_l3": company_a["category_l3"],
        "location": company_a["site_name"].fillna("UNKNOWN"),
        "classification_method": company_a["classification_method"],
        "confidence": company_a["classification_confidence"],
        "include": company_a["include"]
    })

    # Normalize Company B
    b = pd.DataFrame({
        "company": "Company B",
        "record_id": company_b["line_id"],
        "supplier": company_b["vendor_id"].fillna("UNKNOWN"),
        "spend": pd.to_numeric(company_b["spend_usd"], errors="coerce"),
        "date": pd.to_datetime(company_b["parsed_date"], errors="coerce"),
        "category_l1": company_b["predicted_l1"],
        "category_l2": company_b["predicted_l2"],
        "category_l3": company_b["predicted_l3"],
        "location": company_b["region"].fillna("UNKNOWN"),
        "classification_method": company_b["classification_method"],
        "confidence": company_b["confidence"],
        "include": company_b["include"]
    })

    data = pd.concat([a, b], ignore_index=True)

    data["include"] = (
        data["include"].astype(str).str.lower() == "true"
    )

    for column in ["category_l1", "category_l2", "category_l3"]:
        data[column] = data[column].fillna("UNCLASSIFIED")
        data[column] = data[column].replace("", "UNCLASSIFIED")

    data["supplier"] = data["supplier"].fillna("UNKNOWN")
    data["location"] = data["location"].fillna("UNKNOWN")

    data["is_classified"] = (
        data["category_l3"] != "UNCLASSIFIED"
    )

    return data


try:
    data = load_data()
except Exception as error:
    st.error(f"Unable to load procurement data: {error}")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

company_options = ["Company A", "Company B"]

selected_company = st.sidebar.selectbox(
    "Select Company",
    company_options
)

currency = "ZAR" if selected_company == "Company A" else "USD"
currency_symbol = "R" if currency == "ZAR" else "$"

df = data[
    (data["company"] == selected_company) &
    (data["include"])
].copy()

valid_dates = df["date"].dropna()

if not valid_dates.empty:

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.sidebar.date_input(
        "Transaction date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range

        if start_date == min_date and end_date == max_date:
            pass
        else:
            df = df[
                df["date"].notna() &
                df["date"].dt.date.between(start_date, end_date)
            ]

categories = sorted(df["category_l1"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox(
    "Category L1",
    ["All"] + categories
)

if selected_category != "All":
    df = df[df["category_l1"] == selected_category]

top_n = st.sidebar.slider(
    "Top suppliers to display",
    min_value=5,
    max_value=25,
    value=10
)

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

total_spend = df["spend"].sum()
total_records = len(df)
total_suppliers = df["supplier"].nunique()

classified_records = df["is_classified"].sum()
unclassified_records = total_records - classified_records

classification_rate = (
    classified_records / total_records * 100
    if total_records else 0
)

unclassified_spend = df.loc[
    ~df["is_classified"], "spend"
].sum()

# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------

st.subheader(f"{selected_company} Executive Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Procurement Spend",
    f"{currency_symbol}{total_spend:,.2f}"
)

k2.metric(
    "Transactions",
    f"{total_records:,}"
)

k3.metric(
    "Unique Suppliers",
    f"{total_suppliers:,}"
)

k4.metric(
    "Classification Coverage",
    f"{classification_rate:.2f}%"
)

st.divider()

# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

overview, categories, suppliers, trends, quality = st.tabs([
    "Executive Overview",
    "Category Analytics",
    "Supplier Analytics",
    "Trends & Locations",
    "Data Quality"
])

# --------------------------------------------------
# TAB 1 - EXECUTIVE OVERVIEW
# --------------------------------------------------

with overview:

    st.subheader("Spend Overview")

    left, right = st.columns(2)

    with left:

        category_summary = (
            df.groupby("category_l1", as_index=False)["spend"]
            .sum()
            .sort_values("spend", ascending=False)
        )

        fig = px.pie(
            category_summary,
            names="category_l1",
            values="spend",
            hole=0.45,
            title="Spend Distribution by Category L1"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        monthly = df.dropna(subset=["date"]).copy()

        monthly["month"] = (
            monthly["date"].dt.to_period("M").astype(str)
        )

        monthly_summary = (
            monthly.groupby("month", as_index=False)["spend"]
            .sum()
            .sort_values("month")
        )

        fig = px.line(
            monthly_summary,
            x="month",
            y="spend",
            markers=True,
            title=f"Monthly Procurement Spend ({currency})"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Unclassified Spend",
        f"{currency_symbol}{unclassified_spend:,.2f}"
    )

# --------------------------------------------------
# TAB 2 - CATEGORY ANALYTICS
# --------------------------------------------------

with categories:

    st.subheader("Procurement Category Analysis")

    for level in ["category_l1", "category_l2", "category_l3"]:

        summary = (
            df.groupby(level, as_index=False)["spend"]
            .sum()
            .sort_values("spend", ascending=False)
        )

        fig = px.bar(
            summary.head(20),
            x="spend",
            y=level,
            orientation="h",
            title=f"Spend by {level.upper()}",
            labels={"spend": f"Spend ({currency})"}
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 3 - SUPPLIER ANALYTICS
# --------------------------------------------------

with suppliers:

    st.subheader("Supplier Spend Analysis")

    supplier_summary = (
        df.groupby("supplier", as_index=False)
        .agg(
            total_spend=("spend", "sum"),
            transactions=("record_id", "nunique")
        )
        .sort_values("total_spend", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        supplier_summary,
        x="total_spend",
        y="supplier",
        orientation="h",
        title=f"Top {top_n} Suppliers by Spend"
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        supplier_summary,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# TAB 4 - TRENDS AND LOCATIONS
# --------------------------------------------------

with trends:

    st.subheader("Monthly Spend Trends")

    trend_data = df.dropna(subset=["date"]).copy()

    trend_data["month"] = (
        trend_data["date"].dt.to_period("M").astype(str)
    )

    trend_summary = (
        trend_data.groupby("month", as_index=False)["spend"]
        .sum()
        .sort_values("month")
    )

    fig = px.area(
        trend_summary,
        x="month",
        y="spend",
        title=f"Monthly Spend Trend ({currency})"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional / Site Spend")

    location_summary = (
        df.groupby("location", as_index=False)["spend"]
        .sum()
        .sort_values("spend", ascending=False)
        .head(20)
    )

    fig = px.bar(
        location_summary,
        x="location",
        y="spend",
        title="Spend by Location"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 5 - DATA QUALITY
# --------------------------------------------------

with quality:

    st.subheader("Classification and Data Quality")

    q1, q2, q3 = st.columns(3)

    q1.metric("Classified Records", f"{classified_records:,}")
    q2.metric("Unclassified Records", f"{unclassified_records:,}")
    q3.metric("Classification Coverage", f"{classification_rate:.2f}%")

    st.metric(
        "Unclassified Spend",
        f"{currency_symbol}{unclassified_spend:,.2f}"
    )

    method_summary = (
        df["classification_method"]
        .fillna("UNKNOWN")
        .value_counts()
        .reset_index()
    )

    method_summary.columns = ["Method", "Records"]

    fig = px.pie(
        method_summary,
        names="Method",
        values="Records",
        title="Classification Method Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Confidence")

    confidence = pd.to_numeric(
        df["confidence"], errors="coerce"
    ).dropna()

    if not confidence.empty:

        fig = px.histogram(
            confidence,
            nbins=20,
            title="Classification Confidence Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Procurement Records")

    st.dataframe(
        df[
            [
                "record_id",
                "supplier",
                "spend",
                "date",
                "category_l1",
                "category_l2",
                "category_l3",
                "location"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Procurement Spend Intelligence Engine | "
    "Company currencies are reported separately. "
    "Unclassified records remain included in spend totals."
)
