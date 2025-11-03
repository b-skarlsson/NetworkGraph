import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval
import re

# --- 1. Load main data ---
relations_file = "export (35).csv"
df = pd.read_csv(relations_file)

# --- 2. Load table size data from file ---
sizes_df = pd.read_csv("top_90_percent_tables3.csv")  # ← your file with TableName,CumulativeShare

# --- 3. Helper: convert to PascalCase ---
def to_pascal_case(name: str) -> str:
    parts = re.split(r'[_\s]+', name.strip().lower())
    return ''.join(p.capitalize() for p in parts)

sizes_df["TableName"] = sizes_df["TableName"].apply(to_pascal_case)
size_dict = dict(zip(sizes_df["TableName"], sizes_df["CumulativeShare"]))

# --- 4. Create directed graph from relations ---
G = nx.DiGraph()
for _, row in df.iterrows():
    if pd.isna(row["Relations"]):
        continue
    source_pc = to_pascal_case(row["TableName"])
    try:
        relations = json.loads(row["Relations"])
    except json.JSONDecodeError:
        try:
            relations = literal_eval(row["Relations"])
        except Exception:
            continue
    for rel in relations:
        target = rel.get("RelatedTable")
        if not target or target == row["TableName"]:
            continue
        target_pc = to_pascal_case(target)
        if source_pc in size_dict and target_pc in size_dict:
            G.add_edge(target_pc, source_pc)  # target → source (dependency)

# --- 5. Outgoing-only subgraph from RetailTransactionTable ---
root_table = to_pascal_case("RETAILTRANSACTIONTABLE")

if root_table not in G:
    print(f"⚠️ Table '{root_table}' not found in graph.")
else:
    # Get all nodes reachable following *outgoing* arrows
    descendants = nx.descendants(G, root_table)
    descendants.add(root_table)
    subG = G.subgraph(descendants).copy()

    # --- Draw outgoing subgraph ---
    plt.figure(figsize=(16, 12))
    pos = nx.kamada_kawai_layout(subG, scale=2.0)

    out_degrees = dict(subG.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    node_colors = [
        (1, 1 - (out_degrees.get(n, 0) / max_out), 1 - (out_degrees.get(n, 0) / max_out))
        for n in subG.nodes()
    ]

    # Bubble area proportional to data share
    node_sizes = [30000 * (size_dict.get(n, 0.0001) ** 0.5) for n in subG.nodes()]

    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, edgecolors="gray")
    nx.draw_networkx_labels(subG, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(subG, pos, alpha=0.8, arrows=True, arrowstyle="-|>", connectionstyle="arc3,rad=0.12")

    plt.title(f"Outgoing dependency subgraph from {root_table}", fontsize=14, pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- 6. Storage share analysis ---
    total_share = sum(size_dict.values())
    subgraph_share = sum(size_dict.get(n, 0) for n in subG.nodes())
    percent_of_total = (subgraph_share / total_share) * 100

    print(f"\nNumber of tables reachable (outgoing from {root_table}): {len(subG.nodes())}")
    print(f"Total cumulative share of these tables: {subgraph_share:.6f}")
    print(f"Share of total storage: {percent_of_total:.2f}%")
   
   # --- 7. Save reachable tables with their cumulative share ---
    reachable_data = [
        {"TableName": n, "CumulativeShare": size_dict.get(n, 0)}
        for n in subG.nodes()
    ]
    reachable_df = pd.DataFrame(reachable_data)
    reachable_df.sort_values("CumulativeShare", ascending=False, inplace=True)

    output_file = "reachable_tables_from_retailtransactiontable.csv"
    reachable_df.to_csv(output_file, index=False)

    print(f"\n✅ Saved {len(reachable_df)} tables to '{output_file}'")