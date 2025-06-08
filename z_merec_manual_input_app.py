import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Z-MEREC Weighting", layout="wide")

st.title("Z-MEREC Manual Input Weighting App")

st.markdown("Upload a CSV file with **alternatives as rows and criteria as columns**.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df)

    col_names = df.columns.tolist()
    st.markdown("### Step 1: Define Criteria Types")
    criteria_types = []
    target_values = []

    for col in col_names[1:]:  # Exclude alternative name
        col1, col2 = st.columns([2, 1])
        with col1:
            ctype = st.selectbox(
                f"Criteria Type for **{col}**", ["Benefit", "Cost", "Target"], key=col
            )
        with col2:
            tval = st.number_input(f"Target (if applicable)", value=0.0, step=0.01, key=f"target_{col}")
        criteria_types.append(ctype)
        target_values.append(tval if ctype == "Target" else None)

    # Step 2: Normalize
    st.markdown("### Step 2: Z-MEREC Calculation")
    data = df.iloc[:, 1:].astype(float).values  # numeric values only
    alt_names = df.iloc[:, 0].values
    n_criteria = data.shape[1]

    norm_data = np.zeros_like(data)

    for j in range(n_criteria):
        col = data[:, j]
        if criteria_types[j] == "Benefit":
            norm_data[:, j] = col / np.max(col)
        elif criteria_types[j] == "Cost":
            norm_data[:, j] = np.min(col) / col
        elif criteria_types[j] == "Target":
            T = target_values[j]
            norm_data[:, j] = 1 / (1 + abs(col - T) / (np.max(col) - np.min(col) + 1e-8))

    # Step 3: Removal effect
    removal_scores = []
    full_score = np.sum(norm_data, axis=1)
    for j in range(n_criteria):
        temp = np.delete(norm_data, j, axis=1)
        temp_score = np.sum(temp, axis=1)
        diff = full_score - temp_score
        removal_scores.append(np.mean(diff))

    weights = np.array(removal_scores)
    weights = weights / np.sum(weights)

    # Output weights
    weight_df = pd.DataFrame({
        "Criteria": col_names[1:],
        "Type": criteria_types,
        "Target": [t if t is not None else "-" for t in target_values],
        "Weight": weights
    })

    st.markdown("### Step 3: Results")
    st.dataframe(weight_df)

    # Bar chart
    st.markdown("### Step 4: Criteria Weight Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(weight_df["Criteria"], weight_df["Weight"], color="orange")
    ax.set_ylabel("Weight")
    ax.set_title("Z-MEREC Criteria Weights")
    st.pyplot(fig)

    # Download section
    st.markdown("### Step 5: Download Results")
    csv = weight_df.to_csv(index=False).encode()
    st.download_button("Download Weights CSV", data=csv, file_name="zmerec_weights.csv", mime="text/csv")

    # Download plot
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Weight Plot (PNG)", data=buf.getvalue(), file_name="zmerec_plot.png", mime="image/png")
