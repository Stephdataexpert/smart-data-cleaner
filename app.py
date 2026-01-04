import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Data Cleaner + Charts", layout="centered")
st.title("🧹 Smart Data Cleaner + Charts")
st.caption("Upload CSV, Excel, or text-based PDF files to analyze, clean, and visualize data")

st.info("📄 Note: PDF support is for **text-based PDFs with tables only** (not scanned images).")

uploaded_file = st.file_uploader("📤 Upload your file", type=["csv", "xlsx", "xlsm", "pdf"])

df = None

# -------------------------------
# Function to clean column names
# -------------------------------
def clean_columns(df):
    cols = df.columns.tolist()
    new_cols = []
    seen = {}
    for c in cols:
        if c is None:
            c = "Unnamed"
        c = str(c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        new_cols.append(c)
    df.columns = new_cols
    df = df.dropna(axis=1, how='all')  # drop fully empty columns
    return df

# -------------------------------
# File processing
# -------------------------------
if uploaded_file:
    try:
        # CSV
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df = clean_columns(df)

        # Excel
        elif uploaded_file.name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(uploaded_file)
            df = clean_columns(df)

        # PDF
        elif uploaded_file.name.endswith(".pdf"):
            tables = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables:
                        if table:
                            df_table = pd.DataFrame(table[1:], columns=table[0])
                            df_table = clean_columns(df_table)
                            tables.append(df_table)
            if not tables:
                st.error("❌ No tables found in this PDF. Please upload a text-based PDF with tables.")
                st.stop()
            df = pd.concat(tables, ignore_index=True)
            st.warning("⚠️ PDF tables cleaned: duplicate/missing columns renamed automatically")

        # -------------------------------
        # Dataset Overview
        # -------------------------------
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        # Data Quality Checks
        st.subheader("🚨 Data Quality Checks")
        missing_values = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        col3, col4 = st.columns(2)
        col3.metric("Missing Values", missing_values)
        col4.metric("Duplicate Rows", duplicate_rows)

        # Raw Data Preview
        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df)

        # Cleaning: drop duplicates and empty rows
        st.subheader("✨ Cleaned Data")
        cleaned_df = df.drop_duplicates().dropna()
        st.dataframe(cleaned_df)

        # Cleaning Summary
        st.subheader("📉 Cleaning Summary")
        col5, col6 = st.columns(2)
        col5.metric("Rows After Cleaning", cleaned_df.shape[0])
        col6.metric("Rows Removed", df.shape[0] - cleaned_df.shape[0])

        # Download cleaned data
        csv = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

        # -------------------------------
        # Dynamic Chart Section
        # -------------------------------
        st.subheader("📊 Generate Custom Chart")
        st.info("Select columns for X and Y axes to visualize relationships")

        if cleaned_df.shape[1] >= 2:
            all_columns = cleaned_df.columns.tolist()
            numeric_columns = cleaned_df.select_dtypes(include=['float', 'int']).columns.tolist()

            x_axis = st.selectbox("Select X-axis", all_columns)
            y_axis = st.selectbox("Select Y-axis", numeric_columns)

            chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar"])

            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(8,5))
                if chart_type == "Scatter":
                    sns.scatterplot(data=cleaned_df, x=x_axis, y=y_axis, ax=ax)
                elif chart_type == "Line":
                    sns.lineplot(data=cleaned_df, x=x_axis, y=y_axis, ax=ax)
                elif chart_type == "Bar":
                    sns.barplot(data=cleaned_df, x=x_axis, y=y_axis, ax=ax)

                ax.set_title(f"{chart_type} Chart of {y_axis} vs {x_axis}")
                st.pyplot(fig)
        else:
            st.info("At least 2 columns are required to generate a chart.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
