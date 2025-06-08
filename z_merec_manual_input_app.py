
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Z-MEREC with Manual Input")
st.title("Z-MEREC Objective Weight Calculator")
st.write("Upload a simple Excel file with your decision matrix. Then define each criterion manually.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Uploaded Decision Matrix")
    st.dataframe(df)

    criteria = list(df.columns[1:])
    st.subheader("Define Criterion Types and Targets (if any)")
    criterion_types = {}
    target_values = {}

    for crit in criteria:
        ctype = st.selectbox(f"Type of '{crit}'", ["Benefit", "Cost", "Target"])
        criterion_types[crit] = ctype
        if ctype == "Target":
            target = st.number_input(f"Target value for '{crit}'")
            target_values[crit] = target

    if st.button("Run Z-MEREC"):
        data = df.iloc[:, 1:].astype(float)
        norm = pd.DataFrame(index=df.index, columns=criteria)

        for crit in criteria:
            x = data[crit]
            if criterion_types[crit] == "Benefit":
                norm[crit] = (x - x.min()) / (x.max() - x.min())
            elif criterion_types[crit] == "Cost":
                norm[crit] = (x.max() - x) / (x.max() - x.min())
            elif criterion_types[crit] == "Target":
                d = abs(x - target_values[crit])
                norm[crit] = 1 - d / d.max()

        norm = norm.fillna(0)
        norm[norm == 0] = 0.001

        S = norm.sum(axis=1)

        E = []
        for crit in criteria:
            S_dash = norm.drop(columns=[crit]).sum(axis=1)
            E_j = (S - S_dash).abs().sum()
            E.append(E_j)

        weights = np.array(E) / sum(E)
        result = pd.DataFrame({"Criterion": criteria, "Z-MEREC Weight": weights})

        st.subheader("Z-MEREC Objective Weights")
        st.dataframe(result)

        st.download_button("Download Results as Excel", result.to_excel(index=False), file_name="z_merec_weights.xlsx")
