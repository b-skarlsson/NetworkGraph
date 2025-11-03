import pandas as pd

# --- 1. Läs in filen ---
file_path = r"C:\Users\b-elladjarv\Desktop\NetworkGraph\network_Structure_Retail.py\top_90_percent_tables3.csv"
df = pd.read_csv(file_path)

# --- 2. Sortera tabeller efter CumulativeShare, störst först ---
df_sorted = df.sort_values(by='CumulativeShare', ascending=False).reset_index(drop=True)

# --- 3. Beräkna kumulativ summa av CumulativeShare ---
df_sorted['CumulativeSum'] = df_sorted['CumulativeShare'].cumsum()

# --- 4. Hitta de tabeller som tillsammans motsvarar 90% ---
df_top_90 = df_sorted[df_sorted['CumulativeSum'] <= 0.9]

# --- 5. Spara eller visa ---
output_path = r"C:\Users\b-elladjarv\Desktop\NetworkGraph\network_Structure_Retail.py\top_90_percent_tables_from_CumulativeShare.csv"
df_top_90[['TableName']].to_csv(output_path, index=False)

print(f"✅✅✅✅✅ Klar! Fil sparad till: {output_path}✅✅✅✅✅")
print("Tabeller som tillsammans motsvarar 90% av totalen:")
print(df_top_90[['TableName']])
