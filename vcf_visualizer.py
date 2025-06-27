import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import io
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
st.title("🧬 VCF File Polymorphic SNP Visualizer")

# === Upload ===
vcf_file = st.file_uploader("📂 Upload a VCF file (.vcf or .vcf.gz)", type=["vcf", "gz"])

col1, col2, col3 = st.columns(3)
with col1:
    window_size = st.number_input("Window Size (bp)", min_value=1000, step=1000, value=1_000_000)
with col2:
    step_size = st.number_input("Step Size (bp)", min_value=1000, step=1000, value=500_000)
with col3:
    min_depth = st.number_input("Minimum Depth (DP)", min_value=0, step=1, value=0)

filter_biallelic = st.checkbox("✅ Biallelic sites only", value=True)

# === VCF Parser ===
def parse_vcf(uploaded_file, min_dp=0, biallelic_only=True):
    samples, records = [], []
    with gzip.open(uploaded_file, 'rt') if uploaded_file.name.endswith(".gz") else io.StringIO(uploaded_file.getvalue().decode()) as f:
        for line in f:
            if line.startswith("##"):
                continue
            elif line.startswith("#CHROM"):
                samples = line.strip().split('\t')[9:]
            else:
                fields = line.strip().split('\t')
                chrom, pos, ref, alt = fields[0], int(fields[1]), fields[3], fields[4]
                if biallelic_only and ("," in alt or len(ref) != 1 or len(alt) != 1):
                    continue
                info = dict([kv.split("=") if "=" in kv else (kv, None) for kv in fields[7].split(";")])
                dp = int(info.get("DP", 0))
                if dp < min_dp:
                    continue
                for sname, sample in zip(samples, fields[9:]):
                    gt = sample.split(":")[0] if sample else "./."
                    records.append({"CHROM": chrom, "POS": pos, "Sample": sname, "GT": gt})
    return pd.DataFrame(records)

def summarize_trends(df):
    summary = []
    top_bin = df.groupby("Bin")["SNPs"].sum().idxmax()
    summary.append(f"📌 Highest SNP density in bin: `{top_bin}`")
    top_sample = df.groupby("Sample")["SNPs"].sum().idxmax()
    summary.append(f"🧬 Most polymorphic sample: `{top_sample}`")
    return " ".join(summary)

# === Main Workflow ===
if vcf_file:
    with st.spinner("🧬 Parsing and processing VCF..."):
        df = parse_vcf(vcf_file, min_dp=min_depth, biallelic_only=filter_biallelic)
        df = df[df["GT"].notna() & ~df["GT"].isin(["./.", ".", "0/0", "0|0"])]

        samples = df["Sample"].unique()
        chroms = sorted(df["CHROM"].unique())
        selected_chroms = st.multiselect("Select Chromosomes", chroms, default=chroms)
        selected_samples = st.multiselect("Select Samples", samples, default=samples)

        df = df[df["CHROM"].isin(selected_chroms) & df["Sample"].isin(selected_samples)]

        # === SNP Count in Bins
        matrix_entries = []
        for sample in selected_samples:
            df_sample = df[df["Sample"] == sample]
            for chrom in selected_chroms:
                chrom_df = df_sample[df_sample["CHROM"] == chrom]
                if chrom_df.empty:
                    continue
                max_pos = chrom_df["POS"].max()
                for start in range(0, max_pos + step_size, step_size):
                    end = start + window_size
                    bin_label = f"{chrom}:{start//1_000_000}-{end//1_000_000}Mb"
                    count = chrom_df[(chrom_df["POS"] >= start) & (chrom_df["POS"] < end)].shape[0]
                    matrix_entries.append({"Sample": sample, "Bin": bin_label, "SNPs": count})

        matrix_df = pd.DataFrame(matrix_entries)

        if matrix_df.empty:
            st.warning("⚠️ No polymorphic SNPs found.")
        else:
            heatmap = matrix_df.pivot_table(index="Sample", columns="Bin", values="SNPs", aggfunc='sum').fillna(0)
            def bin_sort_key(label):
                chrom, rng = label.split(":")
                start_mb = int(rng.split("-")[0])
                return (chrom, start_mb)
            heatmap = heatmap.reindex(sorted(heatmap.columns, key=bin_sort_key), axis=1)

            st.subheader("📊 SNP Density Heatmap")
            fig, ax = plt.subplots(figsize=(max(16, len(heatmap.columns) * 0.25), len(selected_samples) * 0.4))
            im = ax.imshow(heatmap.values, aspect="auto", cmap="Reds")
            ax.set_xticks(np.arange(len(heatmap.columns)))
            ax.set_xticklabels(heatmap.columns, rotation=90, fontsize=6)
            ax.set_yticks(np.arange(len(heatmap.index)))
            ax.set_yticklabels(heatmap.index, fontsize=6)
            ax.set_xlabel("Genomic Bins")
            ax.set_ylabel("Samples")
            ax.set_title("Polymorphic SNPs per Window")
            fig.colorbar(im, ax=ax, label="SNP Count")
            st.pyplot(fig)

            st.download_button("⬇️ Download SNP Count Matrix", matrix_df.to_csv(index=False).encode(),
                               file_name="snp_window_counts.csv", mime="text/csv")

            st.markdown("### 🤖 AI Summary")
            st.success(summarize_trends(matrix_df))

            # === PCA
            st.subheader("🧠 PCA Clustering")
            scaled_data = StandardScaler().fit_transform(heatmap)
            pca = PCA(n_components=2)
            pca_df = pd.DataFrame(pca.fit_transform(scaled_data), columns=["PC1", "PC2"])
            pca_df["Sample"] = heatmap.index
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Sample", s=80, ax=ax2)
            st.pyplot(fig2)

            # === Outlier Detection
            with st.expander("🔎 Outlier Detection"):
                iso = IsolationForest(contamination=0.1, random_state=42)
                preds = iso.fit_predict(scaled_data)
                outliers = heatmap.index[preds == -1].tolist()
                if outliers:
                    st.error(f"⚠️ Outliers: {', '.join(outliers)}")
                else:
                    st.success("✅ No major outliers.")

            # === Cosine Similarity
            st.subheader("🔬 Sample Similarity (Cosine Distance)")
            sim_df = pd.DataFrame(cosine_similarity(heatmap), index=heatmap.index, columns=heatmap.index)
            fig_sim, ax_sim = plt.subplots(figsize=(8, 6))
            sns.heatmap(sim_df, cmap="viridis", annot=True, fmt=".2f", ax=ax_sim)
            st.pyplot(fig_sim)

            sim_flat = sim_df.where(np.triu(np.ones(sim_df.shape), 1).astype(bool)).stack()
            top_pairs = sim_flat.sort_values(ascending=False).head(3)
            for (s1, s2), sim in top_pairs.items():
                st.write(f"🔗 `{s1}` ↔ `{s2}` → Score: `{sim:.3f}`")

            # === Dendrogram
            st.subheader("🌿 Sample Clustering Dendrogram")
            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 4))
            dendrogram(linkage(scaled_data, method='ward'), labels=heatmap.index.tolist(), leaf_rotation=90, ax=ax_dendro)
            st.pyplot(fig_dendro)

            # === Signature SNPs (Relaxed Homozygous ALT)
            st.subheader("🧬 Sample-Specific Signature SNPs (Relaxed Homozygous ALT Criteria)")
            sig_snps = []

            grouped = df.groupby(["CHROM", "POS"])
            for (chrom, pos), group in grouped:
                gts = group.set_index("Sample")["GT"].to_dict()
                for sample in samples:
                    sample_gt = gts.get(sample, "./.")
                    if sample_gt != "1/1":
                        continue  # Only consider homozygous ALT

                    other_gts = [gts.get(s, "./.") for s in samples if s != sample]
                    if all(gt != "1/1" for gt in other_gts):
                        sig_snps.append({
                            "Sample": sample,
                            "CHROM": chrom,
                            "POS": pos,
                            "GT": sample_gt
                        })

            if sig_snps:
                sig_snp_df = pd.DataFrame(sig_snps)
                st.success(f"✅ Found {len(sig_snp_df)} relaxed homozygous ALT signature SNPs")
                for sample in sig_snp_df["Sample"].unique():
                    st.markdown(f"#### 🔹 Sample `{sample}`")
                    display = sig_snp_df[sig_snp_df["Sample"] == sample].head(10)
                    st.dataframe(display)

                st.download_button(
                    "⬇️ Download Signature SNPs (CSV)",
                    sig_snp_df.to_csv(index=False).encode(),
                    file_name="signature_snps_relaxed.csv",
                    mime="text/csv"
                )
            else:
                st.info("⚠️ No relaxed homozygous ALT signature SNPs found.")


