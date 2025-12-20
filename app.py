import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import joblib


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Cancer Data Mining & Cleaning",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set(style="whitegrid")


# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #1f2933;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>🧬 Cancer Data Mining & Cleaning App</h1>
    <p>Just for you eng/ Fares &lt;3</p>
</div>
""", unsafe_allow_html=True)


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "Choose Section:",
        [
            "📤 Upload Data",
            "🔍 Explore Data",
            "🧹 Clean Data",
            "📊 Visualize",
            "🧬 Cancer Prediction",
            "💾 Export",
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    if 'data' in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        st.metric("Rows", data.shape[0])
        st.metric("Columns", data.shape[1])
        st.metric("Missing", int(data.isna().sum().sum()))
    else:
        st.info("Upload data to see stats")


# ========== INITIALIZE SESSION STATE ==========
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'show_encoded_preview' not in st.session_state:
    st.session_state.show_encoded_preview = False


# ========== PAGE 1: UPLOAD DATA ==========
if page == "📤 Upload Data":
    st.header("📤 Upload Your Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset in CSV format"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading your data..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.session_state.original_data = data.copy()

                    st.markdown(
                        '<div class="success-box">✅ File loaded successfully!</div>',
                        unsafe_allow_html=True
                    )

                    # Quick preview
                    st.subheader("📋 Quick Preview")
                    st.dataframe(data.head(10), use_container_width=True)

                    # Quick stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📏 Rows", data.shape[0])
                    c2.metric("📊 Columns", data.shape[1])
                    c3.metric("❌ Missing", int(data.isna().sum().sum()))
                    c4.metric(
                        "💾 Size",
                        f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB"
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col2:
        st.info(
            "💡 **Tip:** For cancer data, columns like Age, BMI, Smoking, "
            "GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory, Diagnosis are available."
        )


# ========== PAGE 2: EXPLORE DATA ==========
elif page == "🔍 Explore Data":
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        data = st.session_state.data
        st.header("🔍 Data Exploration")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Overview", "📈 Statistics", "🔢 Data Types", "🎯 Unique Values"]
        )

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Dataset Shape")
                st.info(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")

                st.subheader("🏷️ Column Names")
                st.write(data.columns.tolist())

            with col2:
                st.subheader("👀 Preview Options")
                preview_option = st.radio(
                    "Select view:",
                    ["Head", "Tail", "Sample", "Summary"]
                )
                if preview_option == "Head":
                    n_rows = st.slider("Number of rows:", 1, len(data), 10)
                    st.dataframe(data.head(n_rows), use_container_width=True)
                elif preview_option == "Tail":
                    n_rows = st.slider("Number of rows:", 1, len(data), 10)
                    st.dataframe(data.tail(n_rows), use_container_width=True)
                elif preview_option == "Sample":
                    st.dataframe(
                        data.sample(min(10, len(data))), use_container_width=True
                    )
                else:
                    st.write("**Dataset Description:**")
                    buffer = []
                    buffer.append(
                        f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns"
                    )
                    buffer.append(
                        f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024:.2f} KB"
                    )
                    buffer.append(f"**Duplicates:** {data.duplicated().sum()} rows")
                    buffer.append(
                        f"**Missing Values:** {int(data.isna().sum().sum())} cells"
                    )
                    st.markdown("\n\n".join(buffer))
                    st.write("**Column Info:**")
                    col_info = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes.astype(str),
                        'Non-Null': data.count(),
                        'Unique': [data[col].nunique() for col in data.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)

        with tab2:
            st.subheader("📈 Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)

            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 0:
                selected_col = st.selectbox(
                    "Select column for distribution:",
                    numeric_cols
                )
                fig = px.histogram(
                    data,
                    x=selected_col,
                    nbins=30,
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#6366f1']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("🔢 Data Types")
            dtype_df = pd.DataFrame({
                'Column': data.dtypes.index,
                'Data Type': data.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)

            type_counts = data.dtypes.astype(str).value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Data Types Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("🎯 Unique Values")
            for column in data.columns:
                with st.expander(f"📌 {column}"):
                    unique_count = data[column].nunique()
                    st.write(f"**Unique values:** {unique_count}")
                    if data[column].dtype == 'object' and unique_count < 20:
                        st.write(data[column].value_counts())


# ========== PAGE 3: CLEAN DATA ==========
elif page == "🧹 Clean Data":
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        data = st.session_state.data.copy()
        st.header("🧹 Data Cleaning")

        tab1, tab2, tab3 = st.tabs(
            ["❌ Missing Values", "🗑️ Remove Data", "🔄 Transform Data"]
        )

        # ----- Missing values -----
        with tab1:
            st.subheader("❌ Handle Missing Values")

            missing_data = data.isna().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(data) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]

            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)

                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing Count',
                    title="Missing Values by Column",
                    color='Percentage',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

                col_to_fix = st.selectbox(
                    "Select column to fix:",
                    missing_df['Column'].tolist()
                )

                if data[col_to_fix].dtype in ['float64', 'int64']:
                    strategy = st.radio(
                        "Choose strategy:",
                        ["Mean", "Median", "Mode", "Drop Rows"]
                    )
                    if st.button("Apply Strategy", type="primary"):
                        if strategy == "Mean":
                            value = data[col_to_fix].mean()
                            data[col_to_fix].fillna(value, inplace=True)
                            st.success(f"✅ Filled with mean: {value:.2f}")
                        elif strategy == "Median":
                            value = data[col_to_fix].median()
                            data[col_to_fix].fillna(value, inplace=True)
                            st.success(f"✅ Filled with median: {value:.2f}")
                        elif strategy == "Mode":
                            if not data[col_to_fix].mode().empty:
                                value = data[col_to_fix].mode()[0]
                                data[col_to_fix].fillna(value, inplace=True)
                                st.success(f"✅ Filled with mode: {value}")
                        else:
                            data.dropna(subset=[col_to_fix], inplace=True)
                            st.success("✅ Rows with missing values removed")

                        st.session_state.data = data
                        st.rerun()
                else:
                    strategy = st.radio(
                        "Choose strategy:",
                        ["Mode", "Drop Rows", "Fill with 'Unknown'"]
                    )
                    if st.button("Apply Strategy", type="primary"):
                        if strategy == "Mode":
                            if not data[col_to_fix].mode().empty:
                                value = data[col_to_fix].mode()[0]
                                data[col_to_fix].fillna(value, inplace=True)
                                st.success(f"✅ Filled with mode: {value}")
                        elif strategy == "Drop Rows":
                            data.dropna(subset=[col_to_fix], inplace=True)
                            st.success("✅ Rows with missing values removed")
                        else:
                            data[col_to_fix].fillna("Unknown", inplace=True)
                            st.success("✅ Filled with 'Unknown'")

                        st.session_state.data = data
                        st.rerun()
            else:
                st.success("🎉 No missing values found!")

        # ----- Remove data -----
        with tab2:
            st.subheader("🗑️ Remove Data")

            st.write("**Drop Columns**")
            cols_to_drop = st.multiselect(
                "Select columns to remove:",
                data.columns.tolist()
            )
            if st.button("Drop Columns", type="primary", key="drop_cols"):
                if cols_to_drop:
                    data.drop(columns=cols_to_drop, inplace=True)
                    st.session_state.data = data
                    st.success(f"✅ Removed columns: {', '.join(cols_to_drop)}")
                    st.rerun()

            st.markdown("---")

            st.write("**Drop Duplicates**")
            duplicates = data.duplicated().sum()
            st.info(f"Found {duplicates} duplicate rows")
            if st.button("Remove Duplicates", type="primary", key="drop_dupes"):
                if duplicates > 0:
                    data.drop_duplicates(inplace=True)
                    st.session_state.data = data
                    st.success(f"✅ Removed {duplicates} duplicate rows")
                    st.rerun()

        # ----- Transform data -----
        with tab3:
            st.subheader("🔄 Transform Data")

            categorical_cols = data.select_dtypes(include='object').columns.tolist()
            if categorical_cols:
                st.write("**One-Hot Encoding**")
                cols_to_encode = st.multiselect(
                    "Select categorical columns:",
                    categorical_cols
                )
                if st.button("Apply Encoding", type="primary"):
                    if cols_to_encode:
                        data = pd.get_dummies(
                            data,
                            columns=cols_to_encode,
                            prefix=cols_to_encode
                        )
                        st.session_state.data = data
                        st.session_state.show_encoded_preview = True
                        st.success(
                            f"✅ Encoded columns: {', '.join(cols_to_encode)}"
                        )
                        st.rerun()

                if st.session_state.show_encoded_preview:
                    st.write("**Preview of encoded data:**")
                    preview_rows = st.number_input(
                        "Rows to display:",
                        min_value=1,
                        max_value=len(st.session_state.data),
                        value=5,
                        key="preview_input"
                    )
                    st.dataframe(
                        st.session_state.data.head(preview_rows),
                        use_container_width=True
                    )


# ========== PAGE 4: VISUALIZE ==========
elif page == "📊 Visualize":
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        data = st.session_state.data
        st.header("📊 Data Visualization")

        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include='object').columns.tolist()

        tab1, tab2 = st.tabs(["📈 Plotly Charts", "📉 Seaborn Charts"])

        # ----- Plotly charts -----
        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Histogram (Numeric Data)")
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns found!")
                else:
                    numeric_col = st.selectbox(
                        "Select numeric column:",
                        numeric_cols,
                        key="hist_col"
                    )
                    fig = px.histogram(
                        data,
                        x=numeric_col,
                        nbins=30,
                        title=f"Distribution of {numeric_col}",
                        color_discrete_sequence=['#6366f1']
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Bar Chart (Categorical Data)")
                if len(categorical_cols) == 0:
                    st.warning("No categorical columns found!")
                else:
                    cat_col = st.selectbox(
                        "Select categorical column:",
                        categorical_cols,
                        key="bar_col"
                    )
                    value_counts = data[cat_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Bar Chart - {cat_col}",
                        labels={'x': cat_col, 'y': 'Count'},
                        color_discrete_sequence=['#10b981']
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("🔗 Correlation Analysis")

            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis!")
            else:
                default_target = numeric_cols.index("Diagnosis") if "Diagnosis" in numeric_cols else 0
                target = st.selectbox(
                    "Select target variable:",
                    numeric_cols,
                    index=default_target,
                    key="corr_target"
                )
                correlations = data[numeric_cols].corr()[target].drop(target).sort_values(ascending=False)

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = px.bar(
                        x=correlations.index,
                        y=correlations.values,
                        title=f"Correlation with {target}",
                        labels={'x': 'Features', 'y': 'Correlation'},
                        color=correlations.values,
                        color_continuous_scale='RdBu_r',
                        color_continuous_midpoint=0
                    )
                    fig.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Correlation Values:**")
                    corr_df = pd.DataFrame({
                        'Feature': correlations.index,
                        'Correlation': correlations.values.round(3)
                    })
                    st.dataframe(corr_df, use_container_width=True, height=500)

        # ----- Seaborn charts -----
        with tab2:
            st.subheader("📉 Seaborn-Based Visualizations")

            num_cols = numeric_cols
            cat_cols = categorical_cols

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Histogram for numeric column")
                if num_cols:
                    default_idx = num_cols.index("Age") if "Age" in num_cols else 0
                    num_col = st.selectbox(
                        "Select numeric column",
                        num_cols,
                        index=default_idx,
                        key="sns_hist"
                    )
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(data[num_col], kde=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("No numeric columns found.")

            with c2:
                st.markdown("#### Boxplot: Diagnosis vs categorical")
                if "Diagnosis" in num_cols and cat_cols:
                    default_cat_idx = 0
                    cat_col = st.selectbox(
                        "Select categorical column",
                        cat_cols,
                        index=default_cat_idx,
                        key="sns_box"
                    )
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.boxplot(data=data, x=cat_col, y="Diagnosis", ax=ax2)
                    ax2.tick_params(axis="x", rotation=45)
                    st.pyplot(fig2)
                else:
                    st.write("Need 'Diagnosis' numeric and at least one categorical column.")

            st.markdown("#### Scatter plot: Diagnosis vs numeric feature")
            if "Diagnosis" in num_cols and len(num_cols) > 1:
                other_num_cols = [c for c in num_cols if c != "Diagnosis"]
                default_scatter_idx = 0
                scatter_col = st.selectbox(
                    "Select numeric column (X-axis)",
                    other_num_cols,
                    index=default_scatter_idx,
                    key="sns_scatter"
                )
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=data, x=scatter_col, y="Diagnosis", ax=ax3)
                st.pyplot(fig3)
            else:
                st.write("Not enough numeric columns for scatter plot.")

            st.markdown("#### Countplot for categorical column")
            if cat_cols:
                default_count_idx = 0
                count_col = st.selectbox(
                    "Select categorical column for countplot",
                    cat_cols,
                    index=default_count_idx,
                    key="sns_count"
                )
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=data, x=count_col, ax=ax4)
                ax4.tick_params(axis="x", rotation=45)
                st.pyplot(fig4)
            else:
                st.write("No categorical columns found.")


# ========== PAGE 5: CANCER PREDICTION ==========
elif page == "🧬 Cancer Prediction":
    st.header("🧬 Colorectal Cancer Prediction")
    st.write("Enter patient information to predict the probability of colorectal cancer.")

    # load model once
    if "crc_model" not in st.session_state:
        st.session_state.crc_model = joblib.load("crc_model.pkl")
    model = st.session_state.crc_model

    age = st.number_input("Age", min_value=18, max_value=100, value=50)

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

    smoking = st.selectbox(
        "Smoking",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    genetic_risk = st.selectbox(
        "Genetic Risk",
        options=[0, 1, 2],
        format_func=lambda x: ["Low", "Medium", "High"][x]
    )

    physical_activity = st.slider(
        "Physical Activity (1 = Low, 5 = High)",
        min_value=1,
        max_value=5,
        value=3
    )

    alcohol = st.slider(
        "Alcohol Intake (per week)",
        min_value=0,
        max_value=5,
        value=1
    )

    cancer_history = st.selectbox(
        "Family Cancer History",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    if st.button("🔍 Predict"):
        patient = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'Smoking': [smoking],
            'GeneticRisk': [genetic_risk],
            'PhysicalActivity': [physical_activity],
            'AlcoholIntake': [alcohol],
            'CancerHistory': [cancer_history]
        })

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        st.subheader("📊 Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Cancer\n\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ Low Risk of Cancer\n\nProbability: {probability:.2%}")


# ========== PAGE 6: EXPORT ==========
elif page == "💾 Export":
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        data = st.session_state.data
        st.header("💾 Export Cleaned Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Final Dataset Preview")
            st.dataframe(data, use_container_width=True)

            st.subheader("📊 Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", data.shape[0])
            c2.metric("Total Columns", data.shape[1])
            c3.metric("Missing Values", int(data.isna().sum().sum()))

        with col2:
            st.subheader("⬇️ Download")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ Ready to download!")

        if st.session_state.original_data is not None:
            original = st.session_state.original_data
            st.info(f"""
**Changes Made:**
- Original rows: {original.shape[0]}
- Current rows: {data.shape[0]}
- Rows removed: {original.shape[0] - data.shape[0]}
- Original columns: {original.shape[1]}
- Current columns: {data.shape[1]}
            """)


# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>please give us bonus ❤️ </p>
</div>
""", unsafe_allow_html=True)
