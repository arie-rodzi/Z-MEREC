import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Z-MEREC Decision Support", layout="wide")

st.title("Z-MEREC: Manual Criteria Type and Target-Based Weighting")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df)

    with st.form("criteria_form"):
        st.subheader("Step 1: Define Criteria Type and Targets")
        criteria = df.columns[1:]
        criteria_types = {}
        target_values = {}

        for crit in criteria:
            col1, col2 = st.columns([2, 2])
            with col1:
                ctype = st.selectbox(
                    f"Type for '{crit}'",
                    ["Benefit", "Cost", "Target"],
                    key=f"type_{crit}"
                )
                criteria_types[crit] = ctype
            with col2:
                if ctype == "Target":
                    tval = st.number_input(f"Target value for '{crit}'", key=f"target_{crit}")
                    target_values[crit] = tval

        submitted = st.form_submit_button("Calculate Weights")

    if submitted:
        st.subheader("Step 2: Normalization Based on Criteria Type")

        norm_df = df.copy()
        for crit in criteria:
            col = df[crit].astype(float)
            if criteria_types[crit] == "Benefit":
                norm_df[crit] = (col - col.min()) / (col.max() - col.min())
            elif criteria_types[crit] == "Cost":
                norm_df[crit] = (col.max() - col) / (col.max() - col.min())
            elif criteria_types[crit] == "Target":
                target = target_values[crit]
                norm_df[crit] = 1 - abs(col - target) / (col.max() - col.min())

        st.dataframe(norm_df)

        st.subheader("Step 3: Compute Removal Effects and Weights")

        scores_with_all = norm_df[criteria].mean(axis=1)
        removal_effects = []
        for crit in criteria:
            temp_df = norm_df.drop(columns=crit)
            avg_score = temp_df.mean(axis=1)
            effect = abs(avg_score - scores_with_all).sum()
            removal_effects.append(effect)

        weights = np.array(removal_effects) / sum(removal_effects)
        weight_df = pd.DataFrame({
            "Criteria": criteria,
            "Type": [criteria_types[c] for c in criteria],
            "Weight": weights
        })

        st.dataframe(weight_df)

        st.subheader("Step 4: Visualization")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(weight_df["Criteria"], weight_df["Weight"], color='orange')
        ax.set_title("Z-MEREC Criteria Weights")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Criteria")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        csv = weight_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Weights as CSV", data=csv, file_name="zmerec_weights.csv", mime='text/csv')
