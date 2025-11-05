import pandas as pd
import ast
import matplotlib.pyplot as plt

# --- Läs in filen ---
df = pd.read_csv("all_retail_datapoints_filtered_noduplicates.csv")

# --- Lista över tabeller relaterade till RetailTransactionTable ---
related_tables = [
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
    "RETAILTRANSACTIONTAXTRANS"
]

# --- Hämta RetailTransactionTable storlek per miljö (sista dagen) ---
root_row = df.loc[df["TableName"].str.upper() == "RETAILTRANSACTIONTABLE", "TableSizeMB"]
root_sizes_all = [ast.literal_eval(x)[-1] for x in root_row.values]  # <-- ändrat till sista värdet

ratio_data = {}

for table in related_tables:
    row = df.loc[df["TableName"].str.upper() == table, "TableSizeMB"]
    if row.empty:
        print(f"{table} saknas i filen.")
        continue

    # Ta sista dagen för varje miljö
    table_sizes_all = [ast.literal_eval(x)[-1] for x in row.values]  # <-- ändrat till sista värdet

    # --- Beräkna ratio per miljö, hoppa över om RetailTransactionTable = 0 ---
    ratios = [t / r for t, r in zip(table_sizes_all, root_sizes_all) if r != 0]
    ratio_data[table] = ratios

# --- Rita histogram per tabell ---
plt.figure(figsize=(16, 10))
for i, (table, ratios) in enumerate(ratio_data.items(), start=1):
    plt.subplot(4, 4, i)
    plt.hist(ratios, bins=1000, color="skyblue", edgecolor="black")
    if ratios:  # säkerställ att listan inte är tom
        mean_val = sum(ratios) / len(ratios)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    plt.title(table, fontsize=10)
    plt.xlabel("Ratio vs RetailTransactionTable (last day)")
    plt.ylabel("Count")
    plt.legend(fontsize=8)

plt.tight_layout()
plt.show()
