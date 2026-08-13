
import streamlit as st
import pandas as pd
import plotly.express as px


# 1. Page Configuration
st.set_page_config(
    page_title="Bug Life Cycle Analysis Dashboard",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <h1>Bug Life Cycle Analysis Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# 2. Load & Preprocess Data
@st.cache_data
def load_data():
    # BugReport.csv must be in the same folder as dashboard_app.py
    df = pd.read_csv("BugReport.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Date conversion
    df["Date_Closed"] = pd.to_datetime(
        df["Date_Closed"],
        errors="coerce"
    )

    # Numeric conversion
    df["Resolution_Time_Hours"] = pd.to_numeric(
        df["Resolution_Time_Hours"],
        errors="coerce"
    ).fillna(0)

    # Fill missing categorical values
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


df = load_data()


# 3. Sidebar Filters
st.sidebar.header("Filter Options")

selected_sprint = st.sidebar.multiselect(
    "Select Sprint",
    options=sorted(df["Sprint"].unique()),
    default=sorted(df["Sprint"].unique())
)

selected_module = st.sidebar.multiselect(
    "Select Module",
    options=sorted(df["Module"].unique()),
    default=sorted(df["Module"].unique())
)

selected_priority = st.sidebar.multiselect(
    "Select Priority",
    options=sorted(df["Priority"].unique()),
    default=sorted(df["Priority"].unique())
)


# Apply Filters
filtered_df = df[
    (df["Sprint"].isin(selected_sprint))
    & (df["Module"].isin(selected_module))
    & (df["Priority"].isin(selected_priority))
]


# 4. KPI Report Cards
st.subheader("📊 Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Bugs",
        len(filtered_df)
    )

with col2:
    closed_bugs = (
        filtered_df["Status"].astype(str).str.strip().str.lower() == "closed"
    ).sum()

    st.metric(
        "Closed Bugs",
        closed_bugs
    )

with col3:
    avg_res_time = filtered_df["Resolution_Time_Hours"].mean()

    if pd.isna(avg_res_time):
        avg_res_time = 0

    st.metric(
        "Avg Resolution Time",
        f"{avg_res_time:.2f} hrs"
    )

with col4:
    total_modules = filtered_df["Module"].nunique()

    defect_density = (
        len(filtered_df) / total_modules
        if total_modules > 0
        else 0
    )

    st.metric(
        "Defect Density",
        f"{defect_density:.2f} bugs/module"
    )


st.markdown("---")


# 5. Visualizations

# Row 1
row1_col1, row1_col2 = st.columns(2)


# Bug Status Distribution
with row1_col1:
    st.subheader("Bug Status Distribution")

    if not filtered_df.empty:
        fig_status = px.pie(
            filtered_df,
            names="Status",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        st.plotly_chart(
            fig_status,
            use_container_width=True
        )
    else:
        st.info("No data available.")


# Priority Distribution
with row1_col2:
    st.subheader("Priority Distribution")

    priority_data = (
        filtered_df["Priority"]
        .value_counts()
        .reset_index()
    )

    priority_data.columns = ["Priority", "Count"]

    if not priority_data.empty:
        fig_priority = px.bar(
            priority_data,
            x="Priority",
            y="Count",
            color="Priority",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        st.plotly_chart(
            fig_priority,
            use_container_width=True
        )
    else:
        st.info("No data available.")


# Row 2
row2_col1, row2_col2 = st.columns(2)


# Sprint-wise Bug Distribution
with row2_col1:
    st.subheader("Sprint-wise Bug Distribution")

    sprint_data = (
        filtered_df
        .groupby(["Sprint", "Priority"])
        .size()
        .reset_index(name="Count")
    )

    if not sprint_data.empty:
        fig_sprint = px.bar(
            sprint_data,
            x="Sprint",
            y="Count",
            color="Priority",
            barmode="stack"
        )

        st.plotly_chart(
            fig_sprint,
            use_container_width=True
        )
    else:
        st.info("No data available.")


# Module-wise Defect Distribution
with row2_col2:
    st.subheader("Module-wise Defect Distribution")

    module_data = (
        filtered_df["Module"]
        .value_counts()
        .reset_index()
    )

    module_data.columns = ["Module", "Count"]

    if not module_data.empty:
        fig_module = px.bar(
            module_data,
            x="Count",
            y="Module",
            orientation="h",
            color="Count",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(
            fig_module,
            use_container_width=True
        )
    else:
        st.info("No data available.")


# Row 3
row3_col1, row3_col2 = st.columns(2)


# Monthly Bug Resolution Trend
with row3_col1:
    st.subheader("📈 Monthly Bug Resolution Trend")

    trend_df = filtered_df[
        filtered_df["Status"].astype(str).str.strip().str.lower() == "closed"
    ].copy()

    if (
        not trend_df.empty
        and trend_df["Date_Closed"].notna().any()
    ):
        trend_summary = (
            trend_df
            .dropna(subset=["Date_Closed"])
            .groupby(
                trend_df.dropna(subset=["Date_Closed"])["Date_Closed"].dt.to_period("M")
            )
            .size()
            .reset_index(name="Closed_Count")
        )

        trend_summary["Date_Closed"] = (
            trend_summary["Date_Closed"].astype(str)
        )

        fig_trend = px.line(
            trend_summary,
            x="Date_Closed",
            y="Closed_Count",
            markers=True
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )
    else:
        st.info(
            "No closed bugs with valid dates available for trend chart."
        )


# Developer Performance
with row3_col2:
    st.subheader("👥 Developer Performance")

    dev_data = (
        filtered_df
        .groupby("Assigned_To")["Resolution_Time_Hours"]
        .mean()
        .reset_index()
    )

    if not dev_data.empty:
        fig_dev = px.bar(
            dev_data,
            x="Assigned_To",
            y="Resolution_Time_Hours",
            color="Resolution_Time_Hours",
            color_continuous_scale="Reds",
            title="Avg Resolution Time per Developer (Hrs)"
        )

        st.plotly_chart(
            fig_dev,
            use_container_width=True
        )
    else:
        st.info("No developer data available.")


# 6. Root Cause Analysis
st.subheader("🧩 Root Cause Analysis")

root_data = (
    filtered_df["Root_Cause"]
    .value_counts()
    .reset_index()
)

root_data.columns = ["Root Cause", "Count"]

if not root_data.empty:
    fig_root = px.bar(
        root_data,
        x="Root Cause",
        y="Count",
        color="Count",
        color_continuous_scale="Oranges"
    )

    st.plotly_chart(
        fig_root,
        use_container_width=True
    )
else:
    st.info("No root cause data available.")


# 7. Actionable Insights
st.markdown("---")

st.subheader("Actionable Insights")

st.success(
    f"""
    • Total Bugs: {len(filtered_df)}

    • Average Resolution Time: {avg_res_time:.2f} Hours

    • Modules with higher bug count need additional testing.

    • High Priority (P1/P2) bugs should be resolved first.

    • Developers with longer average resolution time may require workload balancing.

    • Root Cause analysis helps reduce recurring defects.
    """
)


# 8. Raw Data Table
st.markdown("---")

st.subheader("📄 Filtered Bug Records")

st.dataframe(
    filtered_df,
    use_container_width=True
)

