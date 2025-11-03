import pandas as pd
import io
import json
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval
import re

# --- 1. Load main data ---
relations_file = "export (35).csv"
df = pd.read_csv(relations_file)

# --- 2. Load CumulativeShare data directly from text ---
size_data = """TableName,CumulativeShare
INVENTTRANS,0.09219508218552312
RETAILTRANSACTIONPAYMENTTRANS,0.06543642419527503
RETAILTRANSACTIONSALESTRANS,0.0614228730733714
RETAILTRANSACTIONTABLE,0.059010377390228214
SALESLINE,0.04624486783895535
GENERALJOURNALACCOUNTENTRY,0.04072728406672815
TAXTRANS,0.03293453303741018
INVENTTRANSPOSTING,0.027529123778813173
INVENTTRANSORIGIN,0.023979701339669056
TAXTRANSGENERALJOURNALACCOUNTENTRY,0.022999791107445475
CUSTINVOICETRANS,0.022043632264623587
RETAILTRANSACTIONTAXTRANS,0.016638148545729013
CUSTTRANS,0.01652768943664675
LEDGERJOURNALTRANS,0.01645936849671797
SYSOUTGOINGEMAILDATA,0.015356687623303857
SOURCEDOCUMENTLINE,0.014926445126630111
INVENTSETTLEMENT,0.013408629748095851
RETAILEODTRANSACTIONTABLE,0.011854040684042243
GENERALJOURNALENTRY,0.011107035911520097
SALESTABLE,0.010505892392066361
INVENTJOURNALTRANS,0.00969921936095734
ACCOUNTINGDISTRIBUTION,0.009641568491421131
RETAILSALESLINE,0.008721535124343746
INVENTDIM,0.008579510862440333
RETAILEODTRANSACTIONSALESTRANS,0.008431060334396858
SYSDATABASELOG,0.008169935600288752
RETAILTRANSACTIONATTRIBUTETRANS,0.008121712547705535
CUSTINVOICEJOUR,0.007811496440178268
CUSTSETTLEMENT,0.007490024551876007
CREDITCARDAUTHTRANS,0.007100695030081146
INVENTREPORTDIMHISTORY,0.006032519229140753
TAXTRANS_REPORTING,0.005945366046496076
PURCHLINE,0.005706374651950102
INVENTSUM,0.005678970498291713
INVENTVALUEREPORTTMPLINE,0.005631783495955746
DOCUHISTORY,0.005596021513866874
RETAILEODSTATEMENTCONTROLLERLOG,0.0055701613589520275
SALESPARMLINE,0.005482843957362907
SUBLEDGERVOUCHERGENERALJOURNALENTRY,0.004881897508671342
RETAILTRANSACTIONINFOCODETRANS,0.0047365293086221935
MCRSALESLINE,0.0044480887370634496
RETAILTRANSACTIONDISCOUNTTRANS,0.003935675270085578
RETAILEODTRANSACTIONERROR,0.0038917272270887504
SUBLEDGERJOURNALACCOUNTENTRYDISTRIBUTION,0.003764872903022611
LEDGERJOURNALTABLE,0.0037527979116043805
RETAILEODTRANSACTIONPAYMENTTRANS,0.003471862138630267
INVENTTRANSORIGINSALESLINE,0.0034034220624103114
INVENTCOSTTRANS,0.003216421991327121
PRICEDISCTABLE,0.0030544387004775944
PDSREBATECUSTINVOICETRANS,0.003049602001563353
VENDTRANS,0.0028571980557845623
VENDINVOICETRANS,0.0027844901965788198
RETAILLOYALTYCARDREWARDPOINTTRANS,0.0027752092901579175
INVENTJOURNALTABLE,0.002656274466773702
SALESPARMTABLE,0.0026013927278392105
RETAILEODTRANSACTIONAGGREGATIONHEADER,0.0025689330866980873
VENDINVOICEINFOTABLE,0.002431753051350609
CUSTINVOICESALESLINK,0.002419656491787992
PRICEDISCADMTRANS,0.0023109829865766746
SUBLEDGERJOURNALACCOUNTENTRY,0.002300193796153307
RETAILSALESTABLE,0.0021780302818712512
CUSTTRANSIDREF,0.0021539408015050676
RETAILEODTRANSACTIONAGGREGATIONTRANS,0.0021120066965397114
RETAILLOG,0.0020678035888253743
PURCHTABLE,0.002051173296070259
RETAILTRANSACTIONFISCALTRANS,0.0020281511387553125
WHSINVENTRESERVE,0.0019989856442809676
VENDSETTLEMENT,0.001977610313944586
VENDINVOICEINFOLINE,0.0019090130634212337
TAXDOCUMENTJSON,0.0019023383813563024
WHSWORKLINE,0.0018675485483132547
ECORESATTRIBUTEVALUE,0.001833921138830227
RETAILDOCUMENTOPERATION,0.0017975369673354212
DIMENSIONFOCUSBALANCE,0.0017402088338465212
MARKUPTRANS,0.0017330945907895318
RETAILEODTRANSACTIONINFOCODETRANS,0.0016597585591354945
DIMENSIONATTRIBUTEVALUECOMBINATION,0.0016533422177736085
RETAILSTATEMENTVOUCHER,0.001625364512584337
DOCUREF,0.0016219281219602524
VENDPACKINGSLIPTRANS,0.0015547528242554546
CREDITCARDCUST,0.0015374660142339386
INVENTTRANSFERLINE,0.0015045059172564311
TAXWORKREGULATION,0.0014276784886695804
TRANSACTIONLOG,0.0014015821122090673
INVENTTRANSORIGINJOURNALTRANS,0.0013761090869296784
GENERALJOURNALACCOUNTENTRYDIMENSION,0.0013602303500865524
RETAILEODSTATEMENTEVENTLOG,0.0013211750302799328
MCRORDERLINE2PRICEHISTORYREF,0.001291744530713999
MCRPRICEHISTORYREF,0.0012827341685510262
SYSOUTGOINGEMAILTABLE,0.0012701865658606957
DIRPARTYTABLE,0.0012622429060941834
TAXUNCOMMITTED,0.0012584166816219137
RETAILEVENTNOTIFICATIONLOG,0.0012563469570416339
INVENTTRANSFERJOURLINE,0.0012558378349677903
TAXTRANS_W,0.0012504908518865252
CUSTPACKINGSLIPTRANS,0.0012484258823012532
INVENTITEMPRICE,0.001235281296308865
WHSSALESLINE,0.0012207746105140141
SYSENCRYPTIONLOG,0.001218065538935795
RETAILSALESDISCOUNTLINE,0.00120628753853399
TRANSTAXINFORMATION,0.0011980512484384212
CUSTTABLE,0.0011947708679986117
RETAILTRANSACTIONFISCALTRANSEXTENDEDDATA,0.0011774445178513546
VENDINVOICEJOUR,0.0011311510348548663
SOURCEDOCUMENTHEADER,0.001108472814284637
SUBLEDGERJOURNALENTRY,0.001086254176089527
PURCHLINEHISTORY,0.001067465647721922
GENERALJOURNALACCOUNTENTRYSUBLEDGERJOURNALACCOUNTENTRY,0.0010619588033480701
SALESPARMSUBTABLE,0.0010334496062255718
TRANSITDOCUMENTTRANSTAXINFORELATION_IN,0.0010294853721028276
MCRORDEREVENTTABLE,0.0010136517918984535
INVENTTRANSORIGINTRANSFER,0.001005774523418658
TAXDOCUMENTROWTRANSITRELATION,0.0009916731736139416
"""
sizes_df = pd.read_csv(io.StringIO(size_data))

# --- 3. Convert names to PascalCase ---
def to_pascal_case(name: str) -> str:
    parts = re.split(r'[_\s]+', name.strip().lower())
    return ''.join(p.capitalize() for p in parts)

sizes_df["TableName"] = sizes_df["TableName"].apply(to_pascal_case)
size_dict = dict(zip(sizes_df["TableName"], sizes_df["CumulativeShare"]))

# --- 4. Create directed graph and add edges ---
G = nx.DiGraph()
for _, row in df.iterrows():
    source = row["TableName"]
    if pd.isna(row["Relations"]):
        continue
    source_pc = to_pascal_case(source)
    try:
        relations = json.loads(row["Relations"])
    except json.JSONDecodeError:
        try:
            relations = literal_eval(row["Relations"])
        except Exception:
            continue
    for rel in relations:
        target = rel.get("RelatedTable")
        if not target or target == source:
            continue
        target_pc = to_pascal_case(target)
        if source_pc not in size_dict or target_pc not in size_dict:
            continue
        G.add_edge(target_pc, source_pc)

# --- 5. Add isolated nodes ---
for node in size_dict.keys():
    if node not in G:
        G.add_node(node)

# --- 6. Identify isolated nodes and print them ---
isolated_nodes = sorted(
    [(n, size_dict[n]) for n in G.nodes() if G.degree(n) == 0],
    key=lambda x: x[1],
    reverse=True
)
print("\nIsolated tables (not connected to any others):")
for name, share in isolated_nodes:
    print(f"  {name:<50}  {share:.8f}")
print(f"\nTotal isolated tables: {len(isolated_nodes)}\n{'-'*70}\n")

# --- 7. Colors & sizes ---
out_degrees = dict(G.out_degree())
max_out = max(out_degrees.values()) if out_degrees else 1
node_colors = [(1, 1 - (out_degrees.get(n, 0) / max_out), 1 - (out_degrees.get(n, 0) / max_out)) for n in G.nodes()]
min_val, max_val = min(size_dict.values()), max(size_dict.values())
node_sizes = [300 + 1200 * ((size_dict[n] - min_val) / (max_val - min_val)) for n in G.nodes()]


# --- 7. Colors & sizes ---
out_degrees = dict(G.out_degree())
max_out = max(out_degrees.values()) if out_degrees else 1
node_colors = [(1, 1 - (out_degrees.get(n, 0) / max_out), 1 - (out_degrees.get(n, 0) / max_out)) for n in G.nodes()]

# --- Make radius proportional to sqrt(share) ---
min_share, max_share = min(size_dict.values()), max(size_dict.values())
scale_factor = 30000  # adjust if you want bigger/smaller bubbles

node_sizes = [scale_factor * (size_dict[n] ** 0.5) for n in G.nodes()]
# --- 8. Layout ---
connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
pos_main = nx.kamada_kawai_layout(G.subgraph(connected_nodes), scale=3.0)
pos = pos_main.copy()

# Place isolated nodes on the right
x_offset = max(x for x, _ in pos_main.values()) + 0.5
y_positions = list(pos_main.values())
min_y, max_y = (min(y for _, y in y_positions), max(y for _, y in y_positions)) if y_positions else (-1, 1)
y_span = max_y - min_y
for i, (node, _) in enumerate(isolated_nodes):
    pos[node] = (x_offset, max_y - (i / max(1, len(isolated_nodes) - 1)) * y_span)

# --- 9. Arrows and margins ---
node_radii = {n: (s ** 0.5) / 2.0 for n, s in zip(G.nodes(), node_sizes)}
edge_margins = {(u, v): {"min_source_margin": node_radii[u]*1.2, "min_target_margin": node_radii[v]*1.2} for u, v in G.edges()}

# --- 10. Draw ---
plt.figure(figsize=(22, 16))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="gray")
nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")

for (u, v), m in edge_margins.items():
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.8, arrows=True, arrowstyle="-|>",
                           min_source_margin=m["min_source_margin"], min_target_margin=m["min_target_margin"],
                           connectionstyle="arc3,rad=0.12")

plt.title("ERP Table Relationship Graph (arrows = one→many, color = #children, size = CumulativeShare, isolated tables listed)", fontsize=15, pad=20)
plt.axis("off")
plt.tight_layout()
plt.show()

# --- 11. Analyze connected components (excluding isolated tables) ---
connected_subgraph = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
components = list(nx.weakly_connected_components(connected_subgraph))
print(f"\nNumber of connected clusters (excluding isolated tables): {len(components)}")

# Sort clusters by size (largest first)
components = sorted(components, key=len, reverse=True)
for i, comp in enumerate(components, 1):
    print(f"  Cluster {i}: {len(comp)} tables")

# --- 12. Plot each cluster individually ---
for i, comp in enumerate(components, 1):
    subG = G.subgraph(comp).copy()
    plt.figure(figsize=(12, 9))
    pos = nx.kamada_kawai_layout(subG, scale=2.0)
    
    # node colors & sizes (reuse same scaling)
    out_degrees = dict(subG.out_degree())
    node_colors = [(1, 1 - (out_degrees.get(n, 0) / max_out), 1 - (out_degrees.get(n, 0) / max_out)) for n in subG.nodes()]
    node_sizes = [300 + 1200 * ((size_dict[n] - min_val) / (max_val - min_val)) for n in subG.nodes()]
    
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color=node_colors, edgecolors="gray")
    nx.draw_networkx_labels(subG, pos, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(subG, pos, alpha=0.8, arrows=True, arrowstyle="-|>", connectionstyle="arc3,rad=0.12")
    
    plt.title(f"Cluster {i} (size={len(subG.nodes())})", fontsize=14, pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.show()