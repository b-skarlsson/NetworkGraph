import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# --- Load data ---
data = pd.read_csv("all_retail_datapoints_filtered_noduplicates.csv", usecols=["new_id", "TableName", "TableSizeMB"])
data["TableName"] = data["TableName"].str.upper().str.strip()

# --- Define tables ---
main_table = "RETAILTRANSACTIONTABLE"
targets = [
    "CUSTTRANS",
    "RETAILEODTRANSACTIONINFOCODETRANS",
    "RETAILEODTRANSACTIONPAYMENTTRANS",
    "RETAILEODTRANSACTIONSALESTRANS",
    "RETAILEODTRANSACTIONTABLE",
    "RETAILLOYALTYCARDREWARDPOINTTRANS",
    "RETAILTRANSACTIONATTRIBUTETRANS",
    "RETAILTRANSACTIONDISCOUNTTRANS",
    "RETAILTRANSACTIONFISCALTRANS",
    "RETAILTRANSACTIONFISCALTRANSEXTENDEDDATA",
    "RETAILTRANSACTIONINFOCODETRANS",
    "RETAILTRANSACTIONPAYMENTTRANS",
    "RETAILTRANSACTIONSALESTRANS",
    "RETAILTRANSACTIONTAXTRANS",
    "PRICEDISCTABLE"
]

# --- Parse TableSizeMB ---
def parse_array(x):
    try:
        return np.array(ast.literal_eval(x)) if isinstance(x, str) else np.nan
    except:
        return np.nan

data["TableSizeMB"] = data["TableSizeMB"].apply(parse_array)

# --- Filter: Exclude arrays that are all zeros ---
def has_nonzero_period(arr, min_nonzero=2):
    """Returnerar True om arrayen har minst min_nonzero element som inte är noll."""
    if not isinstance(arr, np.ndarray):
        return False
    return np.sum(arr != 0) >= min_nonzero

# --- Classify environments ---
def classify_env(arr_main, threshold=0.05):
    """Return 'Linear' or 'Jump' based on relative differences in main table"""
    diffs = np.diff(arr_main)
    rel_diff = np.abs(diffs / (arr_main[:-1] + 1e-8))
    n_jumps = np.sum(rel_diff > threshold)
    return "Linjär / Gradvis" if n_jumps < 3 else "Hopplika / Batch"

env_types = {}
for env_id in data["new_id"].unique():
    subset = data[(data["new_id"] == env_id) & (data["TableName"] == main_table)]
    if subset.empty or not isinstance(subset["TableSizeMB"].iloc[0], np.ndarray):
        continue
    arr_main = subset["TableSizeMB"].iloc[0]
    
    # --- Skip if main table is all zeros ---
    if not has_nonzero_period(arr_main):
        continue
    
    env_types[env_id] = classify_env(arr_main)

# --- Calculate relative percent change ---
rel_changes = {"Linjär / Gradvis": {}, "Hopplika / Batch": {}}

for cluster_name in rel_changes.keys():
    env_ids = [eid for eid, typ in env_types.items() if typ == cluster_name]
    for target in targets:
        all_rel = []
        for env_id in env_ids:
            subset = data[(data["new_id"] == env_id) & (data["TableName"].isin([main_table, target]))]
            subset = subset[subset["TableSizeMB"].apply(lambda x: isinstance(x, np.ndarray))]
            
            # Kontrollera att både main och target finns
            if main_table not in subset["TableName"].values or target not in subset["TableName"].values:
                continue
            
            arr_main = subset[subset["TableName"] == main_table]["TableSizeMB"].iloc[0]
            arr_target = subset[subset["TableName"] == target]["TableSizeMB"].iloc[0]
            
            # --- Skip if any array is all zeros ---
            if not has_nonzero_period(arr_main) or not has_nonzero_period(arr_target):
                continue

            n = min(len(arr_main), len(arr_target))
            if n < 2:
                continue

            if cluster_name == "Linjär / Gradvis":
                # --- Linear: percent change over full sequence ---
                pct_main = (arr_main[-1] - arr_main[0]) / arr_main[0] * 100
                pct_target = (arr_target[-1] - arr_target[0]) / arr_target[0] * 100
                if abs(pct_main) > 1e-8:
                    rel = pct_target / pct_main
                    all_rel.append(rel)
            else:
                # --- Jump: percent change per detected jump ---
                diffs_main = np.diff(arr_main[:n])
                rel_diff_main = np.abs(diffs_main / (arr_main[:-1] + 1e-8))
                jump_idx = np.where(rel_diff_main > 0.05)[0]  # detect jumps >5%
                for idx in jump_idx:
                    dm_pct = (arr_main[idx + 1] - arr_main[idx]) / arr_main[idx] * 100
                    dt_pct = (arr_target[idx + 1] - arr_target[idx]) / arr_target[idx] * 100
                    if abs(dm_pct) < 1e-8:
                        continue
                    rel_jump = dt_pct / dm_pct
                    all_rel.append(rel_jump)

        rel_changes[cluster_name][target] = np.array(all_rel)

# --- Plot histograms with median ---
bins = np.arange(-2, 2.05, 0.05)
for cluster_name, tables_dict in rel_changes.items():
    num_tables = len(tables_dict)
    cols = 4
    rows = int(np.ceil(num_tables / cols))
    plt.figure(figsize=(16, 4*rows))

    for i, (table, rel_vals) in enumerate(tables_dict.items(), start=1):
        plt.subplot(rows, cols, i)
        if len(rel_vals) > 0:
            plt.hist(rel_vals, bins=bins, color="skyblue", edgecolor="black")
            plt.axvline(np.median(rel_vals), color='red', linestyle='--', label=f"Median: {np.median(rel_vals):.2f}")
        plt.title(table, fontsize=10)
        plt.xlabel("Relativ procentuell förändring jämfört med main")
        plt.ylabel("Antal")
        plt.legend(fontsize=8)

    plt.suptitle(f"Relativ procentuell förändring per tabell – {cluster_name}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
