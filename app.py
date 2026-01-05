import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import plotly.express as px

st.set_page_config(page_title="Smart Schema Intelligence", layout="wide")
st.title("🧠 Smart Schema Intelligence Engine")

# =========================
# Helper Functions
# =========================

def normalize_column(col):
    if col is None or str(col).strip() == "":
        return "unnamed_column"
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9_ ]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col

def resolve_duplicate_columns(columns):
    seen = {}
    resolved = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            resolved.append(col)
        else:
            seen[col] += 1
            resolved.append(f"{col}_{seen[col]}")
    return resolved

def infer_column_type(series):
    if series.dropna().empty:
        return "empty"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.nunique() / len(series) < 0.05:
        return "categorical"
    return "text / mixed"

def schema_inference(df):
    schema = []
    for col in df.columns:
        schema.append({
            "column": col,
            "type": infer_column_type(df[col]),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique())
        })
    return pd.DataFrame(schema)

# =========================
# File Upload
# =========================

uploaded_file = st.file_uploader(
    "📤 Upload CSV, Excel, or text-based PDF",
    type=["csv", "xlsx", "xlsm", "pdf"]
)

if uploaded_file:
    try:
        # -------- Read file --------
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                tables = []
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        tables.extend(table)
            df = pd.DataFrame(tables[1:], columns=tables[0])

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        # -------- Normalize schema --------
        df.columns = resolve_duplicate_columns(
            [normalize_column(c) for c in df.columns]
        )

        # -------- Overview --------
        st.subheader("📊 Dataset Overview")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        # -------- Schema --------
        st.subheader("🧠 Inferred Schema")
        schema_df = schema_inference(df)
        st.dataframe(schema_df, use_container_width=True)

        # -------- Preview --------
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # =========================
        # 📊 CUSTOM INTERACTIVE CHART BUILDER (Plotly)
        # =========================

        st.subheader("📊 Interactive Visualization")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for plotting.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis", all_cols)
            with col2:
                y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols)

            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Bar", "Scatter"]
            )

            # Extra customization
            st.sidebar.subheader("📐 Chart Customization")
            title_text = st.sidebar.text_input("Chart Title", f"{y_col} vs {x_col}")
            x_label_rotation = st.sidebar.slider("X-axis label rotation", 0, 90, 45)
            y_label_rotation = st.sidebar.slider("Y-axis label rotation", 0, 90, 0)
            font_size = st.sidebar.slider("Font size", 8, 24, 12)
            width = st.sidebar.slider("Chart Width", 600, 1400, 900)
            height = st.sidebar.slider("Chart Height", 400, 900, 500)

            if st.button("Generate Chart"):
                if chart_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, title=title_text)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, title=title_text)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=title_text)

                # Update layout
                fig.update_layout(
                    width=width,
                    height=height,
                    font=dict(size=font_size),
                    xaxis=dict(tickangle=x_label_rotation),
                    yaxis=dict(tickangle=y_label_rotation),
                    title=dict(x=0.5)  # Center title
                )

                st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Schema analysis and interactive visualization ready")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
