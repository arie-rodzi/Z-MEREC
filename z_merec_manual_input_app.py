import textwrap

# Recreate the fixed Streamlit app code after code state reset
streamlit_app_code = textwrap.dedent("""
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO

    st.title("Z-MEREC: Objective Weight Calculator")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("### Uploaded Data", df)

        criteria = df.columns[1:]
        st.write("### Define Criterion Types")
        criterion_types = {}
        target_values = {}

        for crit in criteria:
            col1, col2 = st.columns([2, 1])
            with col1:
                crit_type = st.selectbox(f"Type for {crit}", ["Benefit", "Cost", "Target"])
            with col2:
                target_val = None
                if crit_type == "Target":
                    target_val = st.number_input(f"Target value for {crit}", value=float(df[crit].mean()))
            criterion_types[crit] = crit_type
            if crit_type == "Target":
                target_values[crit] = target_val

        if st.button("Calculate Z-MEREC Weights"):
            data = df.copy()
            norm_data = pd.DataFrame(index=data.index)

            for crit in criteria:
                col = data[crit]
                if criterion_types[crit] == "Benefit":
                    norm = (col - col.min()) / (col.max() - col.min())
                elif criterion_types[crit] == "Cost":
                    norm = (col.max() - col) / (col.max() - col.min())
                elif criterion_types[crit] == "Target":
                    target = target_values[crit]
                    dev = abs(col - target)
                    max_dev = dev.max()
                    norm = 1 - dev / max_dev if max_dev != 0 else 1
                norm_data[crit] = norm.replace(0, 0.001)  # Prevent zero drop effect

            S = norm_data.sum(axis=1)
            E = {}

            for crit in criteria:
                S_prime = S - norm_data[crit]
                E[crit] = sum(abs(S - S_prime))

            total_E = sum(E.values())
            weights = {crit: E[crit] / total_E for crit in criteria}
            weights_df = pd.DataFrame(weights.items(), columns=["Criterion", "Weight"]).sort_values(by="Weight", ascending=False)

            st.write("### Objective Weights", weights_df)

            # Plot
            fig, ax = plt.subplots()
            ax.bar(weights_df["Criterion"], weights_df["Weight"])
            ax.set_ylabel("Weight")
            ax.set_title("Z-MEREC Criterion Weights")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Download options
            csv = weights_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Weights CSV", csv, "weights.csv", "text/csv")

            buffer = BytesIO()
            fig.savefig(buffer, format="png")
            st.download_button("Download Chart PNG", buffer.getvalue(), "weights_chart.png", "image/png")
""")

# Save to file
file_path = "/mnt/data/z_merec_app_fixed.py"
with open(file_path, "w") as f:
    f.write(streamlit_app_code)

file_path
