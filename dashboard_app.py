import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Software Defect Tracking System",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <h1>Intelligent Software Defect Tracking System with Resolution Assistance</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. LOAD & PREPROCESS DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("BugReport.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Date conversion
    if "Date_Closed" in df.columns:
        df["Date_Closed"] = pd.to_datetime(
            df["Date_Closed"],
            errors="coerce"
        )

    # Resolution time
    if "Resolution_Time_Hours" in df.columns:

        df["Resolution_Time_Hours"] = pd.to_numeric(
            df["Resolution_Time_Hours"],
            errors="coerce"
        ).fillna(0)

    else:

        df["Resolution_Time_Hours"] = 0

    # Fill missing categorical values
    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns

    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


df = load_data()


# ============================================================
# 3. FIND BUG DESCRIPTION COLUMN
# ============================================================

def find_description_column(dataframe):

    possible_columns = [
        "Bug_Description",
        "Bug Description",
        "Description",
        "BugDescription",
        "Bug_Desc",
        "Description_Text",
        "Summary",
        "Bug_Summary"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    return None


# ============================================================
# 4. MAIN TABS - 5 SECTIONS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "🐞 Bug Identification",
        "🔄 Duplicate Bug Detection",
        "🤖 Resolution Assistance",
        "📄 Bug Records"
    ]
)


# ============================================================
# TAB 1 - DASHBOARD
# ============================================================

with tab1:

    # --------------------------------------------------------
    # Sidebar Filters
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Apply Filters
    # --------------------------------------------------------

    filtered_df = df[
        (df["Sprint"].isin(selected_sprint))
        &
        (df["Module"].isin(selected_module))
        &
        (df["Priority"].isin(selected_priority))
    ]

    # ========================================================
    # KPI
    # ========================================================

    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Bugs",
            len(filtered_df)
        )

    with col2:

        closed_bugs = (
            filtered_df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "closed"
        ).sum()

        st.metric(
            "Closed Bugs",
            closed_bugs
        )

    with col3:

        avg_res_time = (
            filtered_df["Resolution_Time_Hours"].mean()
        )

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

    st.divider()

    # ========================================================
    # ROW 1
    # ========================================================

    row1_col1, row1_col2 = st.columns(2)

    # --------------------------------------------------------
    # Bug Status Distribution
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Priority Distribution
    # --------------------------------------------------------

    with row1_col2:

        st.subheader("Priority Distribution")

        priority_data = (
            filtered_df["Priority"]
            .value_counts()
            .reset_index()
        )

        priority_data.columns = [
            "Priority",
            "Count"
        ]

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

    # ========================================================
    # ROW 2
    # ========================================================

    row2_col1, row2_col2 = st.columns(2)

    # --------------------------------------------------------
    # Sprint-wise Bugs
    # --------------------------------------------------------

    with row2_col1:

        st.subheader(
            "Sprint-wise Bug Distribution"
        )

        sprint_data = (
            filtered_df
            .groupby(
                ["Sprint", "Priority"]
            )
            .size()
            .reset_index(
                name="Count"
            )
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

    # --------------------------------------------------------
    # Module-wise Bugs
    # --------------------------------------------------------

    with row2_col2:

        st.subheader(
            "Module-wise Defect Distribution"
        )

        module_data = (
            filtered_df["Module"]
            .value_counts()
            .reset_index()
        )

        module_data.columns = [
            "Module",
            "Count"
        ]

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

    # ========================================================
    # ROW 3
    # ========================================================

    row3_col1, row3_col2 = st.columns(2)

    # --------------------------------------------------------
    # Monthly Resolution Trend
    # --------------------------------------------------------

    with row3_col1:

        st.subheader(
            "📈 Monthly Bug Resolution Trend"
        )

        trend_df = filtered_df[
            filtered_df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "closed"
        ].copy()

        if (
            not trend_df.empty
            and trend_df["Date_Closed"].notna().any()
        ):

            trend_df = trend_df.dropna(
                subset=["Date_Closed"]
            )

            trend_df["Month"] = (
                trend_df["Date_Closed"]
                .dt.to_period("M")
                .astype(str)
            )

            trend_summary = (
                trend_df
                .groupby("Month")
                .size()
                .reset_index(
                    name="Closed_Count"
                )
            )

            fig_trend = px.line(
                trend_summary,
                x="Month",
                y="Closed_Count",
                markers=True
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True
            )

        else:

            st.info(
                "No closed bugs with valid dates available."
            )

    # --------------------------------------------------------
    # Developer Performance
    # --------------------------------------------------------

    with row3_col2:

        st.subheader(
            "👥 Developer Performance"
        )

        dev_data = (
            filtered_df
            .groupby("Assigned_To")
            ["Resolution_Time_Hours"]
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

            st.info(
                "No developer data available."
            )

    # ========================================================
    # ROOT CAUSE
    # ========================================================

  
    st.subheader(
        "🧩 Root Cause Analysis"
    )

    root_data = (
        filtered_df["Root_Cause"]
        .value_counts()
        .reset_index()
    )

    root_data.columns = [
        "Root Cause",
        "Count"
    ]

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

        st.info(
            "No root cause data available."
        )


    # ========================================================
    # 7. ACTIONABLE INSIGHTS
    # ========================================================

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







# ============================================================
# TAB 2 - BUG IDENTIFICATION
# ============================================================

with tab2:

    st.header("🐞 Bug Identification")

    st.write(
        "Analyze a newly reported software defect and "
        "identify its basic characteristics."
    )

    st.divider()

    st.subheader("📝 New Bug Report")

    col1, col2 = st.columns(2)

    with col1:

        identification_title = st.text_input(
            "Bug Title",
            placeholder="Example: Login button not working",
            key="identification_title"
        )

    with col2:

        identification_module = st.selectbox(
            "Affected Module",
            [
                "Authentication",
                "Checkout",
                "Dashboard",
                "API",
                "Database",
                "Frontend",
                "Other"
            ],
            key="identification_module"
        )

    identification_description = st.text_area(
        "Bug Description",
        height=160,
        placeholder=(
            "Describe what happened, expected behavior "
            "and actual behavior..."
        ),
        key="identification_description"
    )

    col3, col4 = st.columns(2)

    with col3:

        identification_severity = st.selectbox(
            "Severity",
            [
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            key="identification_severity"
        )

    with col4:

        identification_status = st.selectbox(
            "Current Status",
            [
                "New",
                "Open",
                "In Progress",
                "Resolved",
                "Closed"
            ],
            key="identification_status"
        )

    st.write("")

    identify_bug_button = st.button(
        "🔍 Identify Bug",
        type="primary",
        use_container_width=True,
        key="identify_bug_button"
    )

    if identify_bug_button:

        if identification_description.strip() == "":

            st.warning(
                "Please enter a bug description."
            )

        else:

            if identification_severity == "Critical":
                suggested_priority = "P1"

            elif identification_severity == "High":
                suggested_priority = "P2"

            elif identification_severity == "Medium":
                suggested_priority = "P3"

            else:
                suggested_priority = "P4"

            text = (
                identification_title
                + " "
                + identification_description
            ).lower()

            if any(
                word in text
                for word in [
                    "login",
                    "password",
                    "authentication",
                    "sign in",
                    "logout"
                ]
            ):

                bug_category = "Authentication Issue"

            elif any(
                word in text
                for word in [
                    "payment",
                    "checkout",
                    "transaction",
                    "order"
                ]
            ):

                bug_category = "Transaction / Checkout Issue"

            elif any(
                word in text
                for word in [
                    "api",
                    "server",
                    "request",
                    "response",
                    "endpoint"
                ]
            ):

                bug_category = "API / Server Issue"

            elif any(
                word in text
                for word in [
                    "database",
                    "sql",
                    "query",
                    "record"
                ]
            ):

                bug_category = "Database Issue"

            elif any(
                word in text
                for word in [
                    "button",
                    "screen",
                    "display",
                    "layout",
                    "css",
                    "ui"
                ]
            ):

                bug_category = "Frontend / UI Issue"

            else:

                bug_category = "Application Logic Issue"

            st.divider()

            st.subheader(
                "📋 Bug Identification Result"
            )

            result1, result2, result3, result4 = st.columns(4)

            with result1:

                st.metric(
                    "Category",
                    bug_category
                )

            with result2:

                st.metric(
                    "Module",
                    identification_module
                )

            with result3:

                st.metric(
                    "Severity",
                    identification_severity
                )

            with result4:

                st.metric(
                    "Suggested Priority",
                    suggested_priority
                )

            st.divider()

            st.subheader(
                "📌 Bug Summary"
            )

            st.info(
                f"""
                **Bug Title:** {
                    identification_title
                    if identification_title
                    else "Untitled Bug"
                }

                **Category:** {bug_category}

                **Module:** {identification_module}

                **Severity:** {identification_severity}

                **Suggested Priority:** {suggested_priority}

                **Current Status:** {identification_status}
                """
            )

            st.success(
                "Bug identification completed successfully."
            )


# ============================================================
# TAB 3 - DUPLICATE BUG DETECTION
# ============================================================

with tab3:

    st.header("🔄 Duplicate Bug Detection")

    st.write(
        "Compare a newly reported bug with existing bugs to identify duplicate or similar defects."
    )

    st.divider()

    st.subheader("🔍 Duplicate Bug Analysis")

    duplicate_title = st.text_input(
        "Bug Title",
        placeholder="Example: User cannot login",
        key="duplicate_title"
    )

    duplicate_description = st.text_area(
        "Bug Description",
        height=160,
        placeholder=(
            "Enter the complete description of the new bug..."
        ),
        key="duplicate_description"
    )

    st.write("")

    duplicate_button = st.button(
        "🔍 Check for Duplicate",
        type="primary",
        use_container_width=True,
        key="duplicate_button"
    )

    if duplicate_button:

        if duplicate_description.strip() == "":

            st.warning(
                "Please enter a bug description."
            )

        else:

            description_column = find_description_column(df)

            if description_column is None:

                st.error(
                    "Bug description column was not found "
                    "in BugReport.csv."
                )

                st.write(
                    "Available columns:"
                )

                st.write(
                    list(df.columns)
                )

            else:

                existing_bugs = df.copy()

                existing_bugs[description_column] = (
                    existing_bugs[description_column]
                    .fillna("")
                    .astype(str)
                )

                existing_bugs = existing_bugs[
                    existing_bugs[
                        description_column
                    ].str.strip() != ""
                ].copy()

                if existing_bugs.empty:

                    st.warning(
                        "No existing bug descriptions are "
                        "available for comparison."
                    )

                else:

                    vectorizer = TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2)
                    )

                    existing_vectors = (
                        vectorizer.fit_transform(
                            existing_bugs[
                                description_column
                            ]
                        )
                    )

                    new_bug_vector = (
                        vectorizer.transform(
                            [duplicate_description]
                        )
                    )

                    scores = cosine_similarity(
                        new_bug_vector,
                        existing_vectors
                    )[0]

                    existing_bugs["Similarity"] = scores

                    top_matches = (
                        existing_bugs
                        .sort_values(
                            "Similarity",
                            ascending=False
                        )
                        .head(5)
                    )

                    best_match = top_matches.iloc[0]

                    best_score = (
                        best_match["Similarity"] * 100
                    )

                    st.divider()

                    st.subheader(
                        "🔎 Duplicate Analysis Result"
                    )

                    if best_score >= 70:

                        detection = "Possible Duplicate"

                        st.error(
                            f"⚠️ Possible Duplicate Bug\n\n"
                            f"Similarity: {best_score:.2f}%"
                        )

                    elif best_score >= 40:

                        detection = "Similar Bug"

                        st.warning(
                            f"⚠️ Similar Bug Found\n\n"
                            f"Similarity: {best_score:.2f}%"
                        )

                    else:

                        detection = "No Strong Duplicate"

                        st.success(
                            f"✅ No Strong Duplicate Found\n\n"
                            f"Similarity: {best_score:.2f}%"
                        )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Detection",
                            detection
                        )

                    with col2:

                        st.metric(
                            "Similarity",
                            f"{best_score:.2f}%"
                        )

                    with col3:

                        if "Bug_ID" in best_match.index:

                            st.metric(
                                "Closest Bug",
                                str(
                                    best_match["Bug_ID"]
                                )
                            )

                        else:

                            st.metric(
                                "Closest Bug",
                                "N/A"
                            )

                    with col4:

                        if "Module" in best_match.index:

                            st.metric(
                                "Module",
                                str(
                                    best_match["Module"]
                                )
                            )

                        else:

                            st.metric(
                                "Module",
                                "N/A"
                            )

                    st.subheader(
                        "📊 Similarity Score"
                    )

                    st.progress(
                        min(best_score / 100, 1.0)
                    )

                    st.caption(
                        f"{best_score:.2f}% similarity"
                    )

                    st.divider()

                    st.subheader(
                        "🔍 Closest Existing Bug"
                    )

                    if "Bug_ID" in best_match.index:

                        st.write(
                            f"**Bug ID:** "
                            f"{best_match['Bug_ID']}"
                        )

                    st.write(
                        f"**Description:** "
                        f"{best_match[description_column]}"
                    )

                    if "Module" in best_match.index:

                        st.write(
                            f"**Module:** "
                            f"{best_match['Module']}"
                        )

                    if "Severity" in best_match.index:

                        st.write(
                            f"**Severity:** "
                            f"{best_match['Severity']}"
                        )

                    if "Priority" in best_match.index:

                        st.write(
                            f"**Priority:** "
                            f"{best_match['Priority']}"
                        )

                    if "Status" in best_match.index:

                        st.write(
                            f"**Status:** "
                            f"{best_match['Status']}"
                        )

                    st.divider()

                    st.subheader(
                        "📋 Top 5 Similar Existing Bugs"
                    )

                    display_columns = []

                    for column in [
                        "Bug_ID",
                        description_column,
                        "Module",
                        "Severity",
                        "Priority",
                        "Status",
                        "Root_Cause"
                    ]:

                        if column in top_matches.columns:

                            display_columns.append(
                                column
                            )

                    result_table = (
                        top_matches[
                            display_columns
                        ].copy()
                    )

                    result_table["Similarity"] = (
                        top_matches["Similarity"] * 100
                    ).round(2)

                    st.dataframe(
                        result_table,
                        use_container_width=True,
                        hide_index=True
                    )


# ============================================================
# TAB 4 - RESOLUTION ASSISTANCE
# ============================================================

with tab4:

    st.header("🤖 Resolution Assistance")

    st.write(
        "Analyze a reported defect and receive recommendations "
        "for severity, priority, root cause and resolution."
    )

    st.divider()

    # ========================================================
    # DEFECT INFORMATION
    # ========================================================

    st.subheader("📝 Defect Information")

    col1, col2 = st.columns(2)

    with col1:

        resolution_title = st.text_input(
            "Bug Title",
            placeholder="Example: Payment transaction failed",
            key="resolution_title"
        )

    with col2:

        resolution_module = st.selectbox(
            "Affected Module",
            [
                "Authentication",
                "Checkout",
                "Dashboard",
                "API",
                "Database",
                "Frontend",
                "Other"
            ],
            key="resolution_module"
        )

    resolution_description = st.text_area(
        "Bug Description",
        height=160,
        placeholder=(
            "Describe the problem in detail..."
        ),
        key="resolution_description"
    )

    resolution_severity = st.selectbox(
        "Severity",
        [
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        key="resolution_severity"
    )

    st.write("")

    resolution_button = st.button(
        "🤖 Analyze & Recommend Resolution",
        type="primary",
        use_container_width=True,
        key="resolution_button"
    )

    # ========================================================
    # RESOLUTION ANALYSIS
    # ========================================================

    if resolution_button:

        if resolution_description.strip() == "":

            st.warning(
                "Please enter a bug description."
            )

        else:

            text = (
                resolution_title
                + " "
                + resolution_description
            ).lower()

            # =================================================
            # PRIORITY RECOMMENDATION
            # =================================================

            if resolution_severity == "Critical":

                recommended_priority = "P1"

            elif resolution_severity == "High":

                recommended_priority = "P2"

            elif resolution_severity == "Medium":

                recommended_priority = "P3"

            else:

                recommended_priority = "P4"

            # =================================================
            # ROOT CAUSE + RESOLUTION
            # =================================================

            if resolution_module == "Authentication":

                root_cause = (
                    "Authentication / Login Validation"
                )

                resolution = (
                    "Check authentication logic, credential "
                    "validation, session management and login "
                    "API responses. Review authentication logs "
                    "and reproduce the issue with valid and "
                    "invalid credentials."
                )

            elif resolution_module == "Checkout":

                root_cause = (
                    "Payment / Transaction Processing Issue"
                )

                resolution = (
                    "Check payment gateway integration, "
                    "transaction validation, payment API "
                    "responses and order-status update logic. "
                    "Review server and payment gateway logs."
                )

            elif resolution_module == "Database":

                root_cause = (
                    "Database / Query Issue"
                )

                resolution = (
                    "Check database connectivity, SQL queries, "
                    "data validation, transactions and database "
                    "logs."
                )

            elif resolution_module == "API":

                root_cause = (
                    "API / Server Issue"
                )

                resolution = (
                    "Check API endpoint, request and response "
                    "handling, server logs, timeout settings "
                    "and server availability."
                )

            elif resolution_module == "Frontend":

                root_cause = (
                    "Frontend / UI Issue"
                )

                resolution = (
                    "Check frontend components, event handlers, "
                    "JavaScript errors, CSS and browser console "
                    "logs."
                )

            elif resolution_module == "Dashboard":

                root_cause = (
                    "Dashboard / Data Visualization Issue"
                )

                resolution = (
                    "Check dashboard data loading, filtering, "
                    "chart configuration, API responses and "
                    "frontend components."
                )

            elif any(
                word in text
                for word in [
                    "slow",
                    "delay",
                    "performance",
                    "loading"
                ]
            ):

                root_cause = (
                    "Performance Issue"
                )

                resolution = (
                    "Check database queries, API response time, "
                    "application performance and resource "
                    "utilization."
                )

            elif any(
                word in text
                for word in [
                    "button",
                    "screen",
                    "display",
                    "layout",
                    "ui",
                    "css"
                ]
            ):

                root_cause = (
                    "Frontend / UI Issue"
                )

                resolution = (
                    "Check UI components, event handlers, "
                    "browser console errors and frontend logic."
                )

            else:

                root_cause = (
                    "Application Logic Issue"
                )

                resolution = (
                    "Review the affected application logic, "
                    "input validation and processing flow. "
                    "Check application logs and reproduce "
                    "the reported defect."
                )

            # =================================================
            # DEFECT ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🧠 Defect Analysis"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Severity",
                    resolution_severity
                )

            with col2:

                st.metric(
                    "Recommended Priority",
                    recommended_priority
                )

            with col3:

                st.metric(
                    "Affected Module",
                    resolution_module
                )

            # =================================================
            # ROOT CAUSE ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🔍 Root Cause Analysis"
            )

            st.info(
                f"**Identified Root Cause:** {root_cause}"
            )

            # =================================================
            # RECOMMENDED RESOLUTION
            # =================================================

            st.subheader(
                "💡 Recommended Resolution"
            )

            st.success(
                resolution
            )

# ============================================================
# TAB 5 - BUG RECORDS
# ============================================================

with tab5:

    st.header("📄 Bug Records")

    st.write(
        "View and search all available software defect records."
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_text = st.text_input(
        "🔎 Search Bug Records",
        placeholder=(
            "Search by Bug ID, Module, Status, Priority, "
            "Root Cause or Description..."
        )
    )

    records_df = df.copy()

    if search_text.strip():

        search_value = search_text.lower()

        mask = (
            records_df
            .astype(str)
            .apply(
                lambda column:
                column.str.lower().str.contains(
                    search_value,
                    na=False
                )
            )
            .any(axis=1)
        )

        records_df = records_df[mask]

    # --------------------------------------------------------
    # Record Metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            len(records_df)
        )

    with col2:

        if "Status" in records_df.columns:

            open_count = (
                records_df["Status"]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "open",
                        "new",
                        "in progress"
                    ]
                )
                .sum()
            )

        else:

            open_count = 0

        st.metric(
            "Open / Active Bugs",
            open_count
        )

    with col3:

        if "Priority" in records_df.columns:

            high_priority = (
                records_df["Priority"]
                .astype(str)
                .str.upper()
                .isin(
                    [
                        "P1",
                        "P2"
                    ]
                )
                .sum()
            )

        else:

            high_priority = 0

        st.metric(
            "High Priority Bugs",
            high_priority
        )

    st.divider()

    # --------------------------------------------------------
    # Data Table
    # --------------------------------------------------------

    st.subheader(
        "📋 Bug Report Data"
    )

    st.dataframe(
        records_df,
        use_container_width=True,
        height=550,
        hide_index=True
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    csv_data = records_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Bug Records",
        data=csv_data,
        file_name="filtered_bug_records.csv",
        mime="text/csv",
        use_container_width=True
    )