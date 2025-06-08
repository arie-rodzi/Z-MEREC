import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Z-MEREC Objective Weight Calculator")

# File upload
uploaded_file = st.file_uploader("Upload Excel file with stock data", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    with st.form("criteria_form"):
        st.write("### Define Criteria Types and Target Values")

        criteria_types = {}
        target_values = {}

        for col in df.columns[1:]:
            c_type = st.selectbox(f"Criterion type for '{col}'", ["Benefit", "Cost", "Target"], key=f"type_{col}")
            criteria_types[col] = c_type
            if c_type == "Target":
                target_values[col] = st.number_input(f"Target value for '{col}'", key=f"target_{col}")

        submitted = st.form_submit_button("Calculate Weights")

    if submitted:
        data = df.copy()
        criteria = data.columns[1:]
        norm_data = pd.DataFrame(index=data.index)

        for crit in criteria:
            values = data[crit]
            if criteria_types[crit] == "Benefit":
                norm = (values - values.min()) / (values.max() - values.min())
            elif criteria_types[crit] == "Cost":
                norm = (values.max() - values) / (values.max() - values.min())
            elif criteria_types[crit] == "Target":
                target = target_values[crit]
                deviation = abs(values - target)
                max_dev = deviation.max()
                norm = 1 - deviation / max_dev
            norm_data[crit] = norm.fillna(0)

        S = norm_data.sum(axis=1)
        removal_effects = {}

        for crit in criteria:
            S_removed = norm_data.drop(columns=[crit]).sum(axis=1)
            removal_effects[crit] = np.sum(abs(S - S_removed))

        total_effect = sum(removal_effects.values())
        weights = {crit: val / total_effect for crit, val in removal_effects.items()}

        st.write("### Calculated Weights")
        weight_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        st.dataframe(weight_df)

        # Plot
        fig, ax = plt.subplots()
        weight_df.sort_values(by='Weight', ascending=False).plot(kind='bar', ax=ax, legend=False)
        ax.set_title("Objective Weights by Z-MEREC")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Criteria")
        st.pyplot(fig)

        # Download links
        st.download_button("Download Weights as CSV", weight_df.to_csv().encode('utf-8'), file_name='z_merec_weights.csv', mime='text/csv')
        fig.savefig("z_merec_weights.png")
        with open("z_merec_weights.png", "rb") as f:
            st.download_button("Download Weight Chart as PNG", f, file_name="z_merec_weights.png")
