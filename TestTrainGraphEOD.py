# -*- coding: utf-8 -*-
"""
Driver ↔ related tables (spread network with R² + significance)
---------------------------------------------------------------
- Builds driver → related table graph
- Fits regressions with bootstrapping
- Saves coefficient and performance CSVs
- Plots network:
    • Edges: Black = significant (with mean ± CI), Grey = uncertain
    • Nodes: Red = drivers, Blue = R² value (dark→light)
    • Node size = importance from top_90_percent_tables3.csv
    • Layout fully spread out (no overlap)
- Plots total MB validation error
"""

import os
import json
import random
import warnings
from ast import literal_eval
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DRIVERS = [
    "RETAILTRANSACTIONTABLE",
    "RETAILTRANSACTIONSALESTRANS",
    "SALESTABLE",
    "SALESLINE",
]

RELATIONS_FILE = "export (35).csv"
SIZES_FILE = "top_90_percent_tables3.csv"
ENV_FILE_1 = "retaileod_envs_part1.csv"
ENV_FILE_2 = "retaileod_envs_part2.csv"

TOP_N = 30
TRAIN_RATIO = 0.7
MIN_POINTS = 10
LOG1P = True
BOOTSTRAP_B = 80
RANDOM_SEED = 42
OUT_DIR = "out"
NETWORK_JITTER = 0.15
NODE_SCALE = 3000.0


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().upper().replace(" ", "").replace("_", "")


def parse_arr(s) -> Optional[np.ndarray]:
    if isinstance(s, str):
        st = s.strip()
        if st.startswith("[") and st.endswith("]"):
            st = st[1:-1]
        try:
            arr = np.fromstring(st, sep=",", dtype=float)
            return arr if arr.size > 0 else None
        except Exception:
            return None
    return None


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------
# LOAD GRAPH
# -------------------------------------------------
def load_driver_graph(top_n: int, drivers_norm: List[str]) -> Tuple[nx.DiGraph, Dict[str, float]]:
    sizes_df = pd.read_csv(SIZES_FILE)
    sizes_df["TableNameNorm"] = sizes_df["TableName"].apply(normalize)
    sizes_df = sizes_df.sort_values("CumulativeShare", ascending=False).head(top_n)
    top_names = set(sizes_df["TableNameNorm"])
    allowed_nodes = top_names | set(drivers_norm)
    size_dict = dict(zip(sizes_df["TableNameNorm"], sizes_df["CumulativeShare"]))

    rel_df = pd.read_csv(RELATIONS_FILE)
    G = nx.DiGraph()

    for _, row in rel_df.iterrows():
        src = normalize(row.get("TableName"))
        if pd.isna(row.get("Relations")) or not src:
            continue
        try:
            rels = json.loads(row["Relations"])
        except Exception:
            try:
                rels = literal_eval(row["Relations"])
            except Exception:
                continue
        if not isinstance(rels, list):
            continue

        for rel in rels:
            tgt = normalize(rel.get("RelatedTable"))
            if not tgt or tgt == src:
                continue
            if (src in drivers_norm or tgt in drivers_norm) and (src in allowed_nodes) and (tgt in allowed_nodes):
                driver = src if src in drivers_norm else tgt
                other = tgt if src in drivers_norm else src
                G.add_edge(driver, other)

    return G, size_dict


# -------------------------------------------------
# LOAD ENV DATA
# -------------------------------------------------
def load_env_averages() -> pd.DataFrame:
    df1 = pd.read_csv(ENV_FILE_1)
    df2 = pd.read_csv(ENV_FILE_2)
    df = pd.concat([df1, df2], ignore_index=True)
    df["TableNameNorm"] = df["TableName"].astype(str).apply(normalize)

    def arr_to_mean(s):
        arr = parse_arr(s)
        if arr is None or len(arr) == 0:
            return None
        return float(np.nanmean(arr))

    df["val"] = df["TableSizeMB"].apply(arr_to_mean)
    df = df.dropna(subset=["val"])
    return df


# -------------------------------------------------
# COLLECT X, y
# -------------------------------------------------
def collect_Xy_for_target(df_env: pd.DataFrame, env_ids: List,
                          drivers_norm: List[str], target_norm: str,
                          min_points: int, log1p: bool):
    env_subset = df_env[df_env["new_id"].isin(env_ids)]
    pivot = env_subset.pivot_table(index="new_id", columns="TableNameNorm", values="val")

    needed_cols = drivers_norm + [target_norm]
    if not all(c in pivot.columns for c in needed_cols):
        return None, None, 0, []

    X = pivot[drivers_norm].values
    y = pivot[target_norm].values
    ids = pivot.index.to_numpy()

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, ids = X[mask], y[mask], ids[mask]
    if len(y) < min_points:
        return None, None, 0, []

    if log1p:
        mask = (y > 0) & np.all(X > 0, axis=1)
        X, y, ids = X[mask], y[mask], ids[mask]
        X, y = np.log1p(X), np.log1p(y)

    return X, y, len(y), ids


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_outdir(OUT_DIR)
    drivers_norm = [normalize(t) for t in DRIVERS]

    # 1) Graph
    G, size_dict = load_driver_graph(TOP_N, drivers_norm)
    connected_targets = sorted({tgt for _, tgt in G.edges()})
    print(f"✅ Found {len(connected_targets)} related tables from drivers (TOP_N={TOP_N}).")

    # 2) Env data
    df_env = load_env_averages()
    envs = df_env["new_id"].unique().tolist()
    random.shuffle(envs)
    n_train = int(TRAIN_RATIO * len(envs))
    train_envs, val_envs = envs[:n_train], envs[n_train:]
    print(f"Training on {len(train_envs)} envs, validating on {len(val_envs)} envs.")

    edge_rows, perf_rows = [], []
    val_actual_totals = {env: 0.0 for env in val_envs}
    val_pred_totals = {env: 0.0 for env in val_envs}

    # 3) Regression
    for target in connected_targets:
        X_train, y_train, _, _ = collect_Xy_for_target(df_env, train_envs, drivers_norm, target, MIN_POINTS, LOG1P)
        X_val, y_val, n_val_rows, val_ids = collect_Xy_for_target(df_env, val_envs, drivers_norm, target, MIN_POINTS, LOG1P)
        if X_train is None or X_val is None:
            continue

        base = LinearRegression().fit(X_train, y_train)
        boot_coefs = [LinearRegression().fit(*resample(X_train, y_train)).coef_ for _ in range(BOOTSTRAP_B)]
        boot_coefs = np.vstack(boot_coefs)
        means, ci90 = boot_coefs.mean(axis=0), 1.64 * boot_coefs.std(axis=0, ddof=1)

        y_pred = base.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        perf_rows.append({"Target": target, "R2_val": r2, "RMSE_val": rmse, "n_val": int(n_val_rows)})

        sig_any = False
        for j, src in enumerate(drivers_norm):
            mean, ci = float(means[j]), float(ci90[j])
            sig = (mean - ci > 0) or (mean + ci < 0)
            if sig:
                sig_any = True
            if G.has_edge(src, target):
                edge_rows.append({"Source": src, "Target": target, "Mean": mean,
                                  "CI90": ci, "Significant": sig,
                                  "n_train": len(X_train), "n_val": n_val_rows})

        if sig_any:
            for env_id, actual, pred in zip(val_ids, y_val, y_pred):
                val_actual_totals[env_id] += actual
                val_pred_totals[env_id] += pred

    # 4) Save results
    edges_df = pd.DataFrame(edge_rows)
    results_df = pd.DataFrame(perf_rows)
    edges_df.to_csv(os.path.join(OUT_DIR, "edge_effects_drivers.csv"), index=False)
    results_df.to_csv(os.path.join(OUT_DIR, "target_metrics.csv"), index=False)

    # -------------------------------------------------
    # 5) Plot network — black/grey edges, labeled, blue nodes by R²
    # -------------------------------------------------
    try:
        sub_nodes = set(drivers_norm) | set(connected_targets)
        subG = G.subgraph(sub_nodes).copy()

        for u, v in list(subG.edges()):
            row = edges_df[(edges_df["Source"] == u) & (edges_df["Target"] == v)]
            if len(row) > 0:
                subG[u][v]["sig"] = bool(row["Significant"].iloc[0])
                subG[u][v]["mean"] = float(row["Mean"].iloc[0])
                subG[u][v]["ci"] = float(row["CI90"].iloc[0])
            else:
                subG[u][v]["sig"] = False
                subG[u][v]["mean"] = 0.0
                subG[u][v]["ci"] = 0.0

        # layout: more spread out
        pos = nx.spring_layout(subG, k=6.0, iterations=600, seed=RANDOM_SEED)
        pos_arr = np.array(list(pos.values()))
        pos_arr += np.random.uniform(-NETWORK_JITTER, NETWORK_JITTER, pos_arr.shape)
        pos = {n: tuple(p) for n, p in zip(subG.nodes(), pos_arr)}

        # node colors and sizes
        r2_map = {normalize(r["Target"]): r["R2_val"] for _, r in results_df.iterrows()}
        sizes_df = pd.read_csv(SIZES_FILE)
        sizes_df["TableNameNorm"] = sizes_df["TableName"].apply(normalize)
        size_dict = dict(zip(sizes_df["TableNameNorm"], sizes_df["CumulativeShare"]))

        node_colors, node_sizes = [], []
        for n in subG.nodes():
            if n in drivers_norm:
                node_colors.append("#e41a1c")  # red driver
            else:
                r2_val = r2_map.get(n, 0.0)
                node_colors.append(plt.cm.Blues(0.3 + 0.7 * max(0, min(1, r2_val))))
            node_sizes.append(NODE_SCALE * (size_dict.get(n, 0.001) ** 0.5))

        # edges
        edge_colors = ["black" if subG[u][v]["sig"] else "grey" for u, v in subG.edges()]
        edge_widths = [2 if subG[u][v]["sig"] else 0.8 for u, v in subG.edges()]

        plt.figure(figsize=(20, 16))
        nx.draw_networkx_edges(subG, pos, edge_color=edge_colors, width=edge_widths,
                               arrows=True, arrowstyle="-|>", connectionstyle="arc3,rad=0.1", alpha=0.9)
        nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, edgecolors="gray", linewidths=1.5)
        nx.draw_networkx_labels(subG, pos, font_size=9, font_weight="bold")

        # edge labels for significant edges
        edge_labels = {(u, v): f"{subG[u][v]['mean']:.2f}±{subG[u][v]['ci']:.2f}"
                       for u, v in subG.edges() if subG[u][v]["sig"]}
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=7, font_color="black")

        plt.title("Driver → Related Tables\nEdges: Black = significant (with mean±CI), Grey = uncertain\nNode color = R² (blue scale), size = importance")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Network plot skipped: {e}")

    # -------------------------------------------------
    # 6) Plot total MB error for validation environments
    # -------------------------------------------------
    print("\n📈 Calculating total actual vs predicted sizes for all validation environments...\n")

    env_summary = []
    for env_id in val_envs:
        df_demo = df_env[df_env["new_id"] == env_id]
        pivot = df_demo.pivot_table(index="new_id", columns="TableNameNorm", values="val").fillna(0.0)

        total_actual, total_pred, used = 0.0, 0.0, 0
        for target, subdf in edges_df.groupby("Target"):
            subdf = subdf[subdf["Significant"] == True]
            if subdf.empty:
                continue
            pred_val = 0.0
            for _, row in subdf.iterrows():
                src, coef = row["Source"], row["Mean"]
                pred_val += coef * pivot.get(src, pd.Series([0.0], index=[env_id])).iloc[0]
            actual_val = pivot.get(target, pd.Series([np.nan], index=[env_id])).iloc[0]
            if pd.isna(actual_val):
                continue
            total_actual += actual_val
            total_pred += pred_val
            used += 1

        if used > 0 and total_actual > 0:
            diff = total_pred - total_actual
            pct_err = 100 * abs(diff) / total_actual
            env_summary.append({
                "Env": env_id[:6],
                "Actual": total_actual,
                "Pred": total_pred,
                "Diff": diff,
                "ErrorPct": pct_err
            })

    if env_summary:
        env_df = pd.DataFrame(env_summary).sort_values("ErrorPct", ascending=False).reset_index(drop=True)
        avg_err = env_df["ErrorPct"].mean()
        med_err = env_df["ErrorPct"].median()
        print(f"✅ Computed errors for {len(env_df)} environments "
              f"(Avg error: {avg_err:.2f}%, Median: {med_err:.2f}%)")

        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = range(len(env_df))
        width = 0.35

        ax1.bar([i - width/2 for i in x], env_df["Actual"], width, label="Actual (MB)", color="#4CAF50", alpha=0.8)
        ax1.bar([i + width/2 for i in x], env_df["Pred"], width, label="Predicted (MB)", color="#2196F3", alpha=0.8)
        ax1.set_xlabel("Validation Environments")
        ax1.set_ylabel("Total Size (MB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(env_df["Env"], rotation=45, ha="right")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(x, env_df["ErrorPct"], color="red", marker="o", label="Absolute % Error")
        ax2.set_ylabel("Absolute % Error")
        ax2.legend(loc="upper right")

        plt.title("Validation Environments — Total Actual vs Predicted Sizes and % Error")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No environments with valid predictions.")


if __name__ == "__main__":
    main()
