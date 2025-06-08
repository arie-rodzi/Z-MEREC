import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="Z-MEREC Calculator", layout="wide")
st.title("Z-MEREC Objective Weighting Tool")

st.markdown("""
This tool calculates **objective weights** for multiple criteria using the **Improvised MEREC (Z-MEREC)** method. 
You can upload an Excel file containing performance values and define:
- Type of each criterion (Benefit, Cost, or Target)
- Target value (only for Target-Optimal type)

Then, click `Calculate` to compute and visualize the results.
""")

uploaded_file = st.file_uploader("Upload Excel File (First row: Stock Names, Columns: Criteria)", type=[".xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df, use_container_width=True)

    stock_names = df.iloc[:, 0]
    data = df.iloc[:, 1:].copy()

    st.subheader("Step 1: Define Criterion Type")
    criteria = list(data.columns)
    criterion_types = []
    target_values = []

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.markdown("**Criterion**")
        for c in criteria:
            st.text(c)
    with col2:
        st.markdown("**Type**")
        for i, c in enumerate(criteria):
            t = st.selectbox(f"Type of {c}", ["Benefit", "Cost", "Target"], key=f"type_{i}")
            criterion_types.append(t)
    with col3:
        st.markdown("**Target Value (if Target)**")
        for i, c in enumerate(criteria):
            if criterion_types[i] == "Target":
                target = st.number_input(f"Target for {c}", key=f"target_{i}")
            else:
                target = None
            target_values.append(target)

    if st.button("Calculate"):
    # Step 2: Normalization
        norm_data = pd.DataFrame(index=data.index, columns=data.columns)

    for i, c in enumerate(criteria):
        x = data[c].astype(float)

        if criterion_types[i] == "Benefit":
            max_val = x.max()
            norm = x / max_val if max_val != 0 else 0

        elif criterion_types[i] == "Cost":
            min_val = x.min()
            norm = min_val / x.replace(0, np.nan)
            norm = norm.fillna(0)

        elif criterion_types[i] == "Target":
            target = target_values[i]
            d = abs(x - target)
            d_max = d.max()
            norm = 1 - (d / d_max) if d_max != 0 else 1

        norm = norm.replace(0, 0.001)  # avoid zero for stability
        norm_data[c] = norm

    st.subheader("Step 2: Normalized Data")
    st.dataframe(norm_data, use_container_width=True)

        # Step 3: Overall Performance Score
        norm_data = norm_data.apply(pd.to_numeric, errors='coerce')
        S = norm_data.sum(axis=1)

        # Step 4: Recalculate Performance without Each Criterion
        S_prime = pd.DataFrame(index=norm_data.index, columns=criteria)
        for c in criteria:
            S_prime[c] = norm_data.drop(columns=c).sum(axis=1)

        # Step 5: Calculate Removal Effects
        E = {}
        for c in criteria:
            E[c] = np.abs(S - S_prime[c]).sum()

        # Step 6: Normalize Removal Effects into Weights
        total_effect = sum(E.values())
        weights = {c: E[c] / total_effect for c in criteria}

        # Display Final Weights
        st.subheader("Final Objective Weights")
        weight_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weight_df = weight_df.sort_values(by='Weight', ascending=False)
        st.dataframe(weight_df.style.format("{:.4f}"))

        # Chart
        fig, ax = plt.subplots()
        ax.bar(weight_df.index, weight_df['Weight'])
        ax.set_ylabel("Weight")
        ax.set_title("Criterion Importance Weights")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # CSV Download
        def convert_df(df):
            return df.to_csv(index=True).encode('utf-8')

        csv = convert_df(weight_df)
        st.download_button(
            label="Download Weights as CSV",
            data=csv,
            file_name='z_merec_weights.csv',
            mime='text/csv',
        )
