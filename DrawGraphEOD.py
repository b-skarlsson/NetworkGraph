import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np

# --- Parameters ---
top_n = 20
jitter_strength = 0.08  # small random offset to separate overlapping nodes

# --- 1️⃣ Load relationship data ---
relations_file = "export (35).csv"
df = pd.read_csv(relations_file)

# --- 2️⃣ Load and trim top N tables ---
sizes_df = pd.read_csv("top_90_percent_tables3.csv")
sizes_df = sizes_df.sort_values("CumulativeShare", ascending=False).head(top_n)

# --- 3️⃣ Normalize table names ---
def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().upper().replace(" ", "").replace("_", "")

df["TableNameNorm"] = df["TableName"].apply(normalize)
sizes_df["TableNameNorm"] = sizes_df["TableName"].apply(normalize)

# --- 4️⃣ Build graph with self-loops removed ---
top_names = set(sizes_df["TableNameNorm"])
size_dict = dict(zip(sizes_df["TableNameNorm"], sizes_df["CumulativeShare"]))

G = nx.DiGraph()

for _, row in df.iterrows():
    if pd.isna(row["Relations"]):
        continue

    src = normalize(row["TableName"])
    if src not in top_names:
        continue

    try:
        rels = json.loads(row["Relations"])
    except json.JSONDecodeError:
        try:
            rels = literal_eval(row["Relations"])
        except Exception:
            continue

    for rel in rels:
        tgt = rel.get("RelatedTable")
        if not tgt:
            continue
        tgt = normalize(tgt)

        # 🚫 Skip self-pointers and non-top tables
        if tgt == src or tgt not in top_names:
            continue

        G.add_edge(tgt, src)

# --- 5️⃣ Define focus tables ---
focus_tables_raw = ["RetailTransactionSalesTrans", "RetailTransactionTable", "SalesTable", "SalesLine"]
focus_tables = [normalize(t) for t in focus_tables_raw]

# --- 6️⃣ Collect connected nodes ---
connected_nodes = set(focus_tables)
for ft in focus_tables:
    if ft in G:
        connected_nodes |= set(G.predecessors(ft))
        connected_nodes |= set(G.successors(ft))

subG = G.subgraph(connected_nodes).copy()

# --- 7️⃣ Draw the subgraph ---
if len(subG.nodes) == 0:
    print("\n⚠️ No connections found — try increasing top_n or check table names.")
else:
    min_share, max_share = min(size_dict.values()), max(size_dict.values())
    scale_factor = 30000
    node_sizes = [scale_factor * (size_dict.get(n, min_share) ** 0.5) for n in subG.nodes()]
    node_colors = ["red" if n in focus_tables else "white" for n in subG.nodes()]

    pos = nx.kamada_kawai_layout(subG, scale=2.5)

    # --- Apply small random jitter to avoid overlapping nodes ---
    pos_array = np.array(list(pos.values()))
    if len(pos_array) > 1:
        pos_array += np.random.uniform(-jitter_strength, jitter_strength, pos_array.shape)
        pos = {n: tuple(p) for n, p in zip(subG.nodes(), pos_array)}

    # --- Compute node radii for arrow margins ---
    node_radii = {n: (s ** 0.5) / 2.0 for n, s in zip(subG.nodes(), node_sizes)}
    edge_margins = {
        (u, v): {
            "min_source_margin": node_radii[u] * 1.2,
            "min_target_margin": node_radii[v] * 1.2
        }
        for u, v in subG.edges()
    }

    # --- Draw nodes ---
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(
        subG, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="gray",
        linewidths=0.8,
        alpha=1.0
    )
    nx.draw_networkx_labels(subG, pos, font_size=8, font_weight="bold")

    # --- Draw edges (curved, offset from nodes) ---
    for (u, v), m in edge_margins.items():
        nx.draw_networkx_edges(
            subG, pos,
            edgelist=[(u, v)],
            alpha=0.8,
            arrows=True,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            min_source_margin=m["min_source_margin"],
            min_target_margin=m["min_target_margin"],
            edge_color="black"
        )


    print(f"\n✅ Nodes in subgraph: {len(subG.nodes)}")
    print(f"✅ Edges in subgraph: {len(subG.edges)}")


    # --- Style ---
    plt.title(
        f"ERP Table Graph (Top {top_n} by Size, Focus on 4 Core Tables)",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    