import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import random
import os

# --- Load data ---
data = pd.read_csv("all_retail_datapoints_filtered.csv", usecols=["new_id", "TableName", "TableSizeMB"])
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

# --- Samla korrelationer per tabell ---
all_corrs = {}
for target in targets:
    pair_tables = [main_table, target]
    subset = data[data["TableName"].isin(pair_tables)]
    pivot = subset.pivot(index="new_id", columns="TableName", values="TableSizeMB").dropna(subset=pair_tables)

    corrs = []
    for arr1, arr2 in zip(pivot[main_table], pivot[target]):
        if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
            n = min(len(arr1), len(arr2))
            if n >= 2:
                c = np.corrcoef(arr1[:n], arr2[:n])[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
    if corrs:
        all_corrs[target] = np.array(corrs)

# --- Rita histogram över korrelationer ---
num_tables = len(all_corrs)
cols = 4
rows = int(np.ceil(num_tables / cols))

plt.figure(figsize=(16, 12))
for i, (table, corrs) in enumerate(all_corrs.items(), start=1):
    plt.subplot(rows, cols, i)
    plt.hist(corrs, bins=np.arange(-1, 1.05, 0.05), color="skyblue", edgecolor="black")
    plt.axvline(np.mean(corrs), color='red', linestyle='--', label=f"Mean: {np.mean(corrs):.2f}")
    plt.title(table, fontsize=10)
    plt.xlabel("Pearson correlation")
    plt.ylabel("Count")
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# --- Skapa mapp för sparade figurer ---
os.makedirs("plots", exist_ok=True)

# --- Funktion för att rita main + alla target-tabeller per environment ---
def plot_envs(env_ids, title):
    if not env_ids:
        return

    rows = len(env_ids)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    fig.suptitle(f"{title}", fontsize=14)

    tables_to_plot = [main_table] + targets
    colors = plt.cm.tab20.colors  # Färgpalett med upp till 20 färger

    for ax, new_id in zip(axes, env_ids):
        subset = data[data["new_id"] == new_id]
        subset = subset[subset["TableName"].isin(tables_to_plot)]
        subset = subset[subset["TableSizeMB"].apply(lambda x: isinstance(x, np.ndarray))]

        for i, table in enumerate(tables_to_plot):
            if table in subset["TableName"].values:
                arr = subset[subset["TableName"] == table]["TableSizeMB"].iloc[0]
                n = len(arr)
                ax.plot(range(n), arr, label=table, color=colors[i % len(colors)])

        # Visa korrelation mot main_table
        corr_texts = []
        for table in targets:
            if table in subset["TableName"].values:
                arr_main = subset[subset["TableName"] == main_table]["TableSizeMB"].iloc[0]
                arr_target = subset[subset["TableName"] == table]["TableSizeMB"].iloc[0]
                n_corr = min(len(arr_main), len(arr_target))
                c = np.corrcoef(arr_main[:n_corr], arr_target[:n_corr])[0, 1]
                if np.isfinite(c):
                    corr_texts.append(f"{table}: {c:.2f}")
        if corr_texts:
            ax.set_title(f"Environment: {new_id}  | Corrs: {', '.join(corr_texts)}")
        else:
            ax.set_title(f"Environment: {new_id}")

        ax.set_ylabel("MB")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Tid (index)")
    plt.tight_layout()
    plt.show()

# --- Loop över targets för att hitta high/mid korrelationer ---
for target, corrs in all_corrs.items():
    pair_tables = [main_table, target]
    subset = data[data["TableName"].isin(pair_tables)]
    pivot = subset.pivot(index="new_id", columns="TableName", values="TableSizeMB").dropna(subset=pair_tables)

    corr_map = {}
    for new_id, row in pivot.iterrows():
        arr1, arr2 = row[main_table], row[target]
        if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
            n = min(len(arr1), len(arr2))
            if n >= 2:
                c = np.corrcoef(arr1[:n], arr2[:n])[0, 1]
                if np.isfinite(c):
                    corr_map[new_id] = c

    if not corr_map:
        continue

    corr_series = pd.Series(corr_map)
    high_corr_ids = corr_series[corr_series >= 0.95].index.tolist()
    mid_corr_ids = corr_series[(corr_series >= 0.55) & (corr_series <= 0.65)].index.tolist()

    high_sample = random.sample(high_corr_ids, min(3, len(high_corr_ids))) if high_corr_ids else []
    mid_sample = random.sample(mid_corr_ids, min(3, len(mid_corr_ids))) if mid_corr_ids else []

    plot_envs(high_sample, "Hög korrelation (≥ 0.95)")
    plot_envs(mid_sample, "Måttlig korrelation (~0.6)")

# --- Extra: Välj några miljöer med flest tabeller och plott main + alla targets ---
env_counts = data.groupby("new_id")["TableName"].nunique()
top_envs = env_counts.sort_values(ascending=False).head(2).index  # t.ex. 2 miljöer

for env_id in top_envs:
    plot_envs([env_id], f"Main + alla target-tabeller för environment {env_id}")
