import pandas as pd
import ast

# --- 1. Läs in filen ---
file_path = r"C:\Users\b-elladjarv\Desktop\NetworkGraph\network_Structure_Retail.py\3cleaned_zero_dips_extended.csv"
df = pd.read_csv(file_path)

# --- 2. Funktion för att ta sista dagens värde från TableSizeMB ---
def last_day_value(x):
    try:
        lst = ast.literal_eval(x)  # Konvertera sträng till lista
        if isinstance(lst, list) and len(lst) > 0:
            return lst[-1]  # sista dagens värde
        else:
            return float(x)
    except:
        return float(x)

df['LastDay'] = df['TableSizeMB'].apply(last_day_value)

# --- 3. Gruppera per TableName och ta summan av sista dagen (om flera rader per tabell) ---
df_grouped = df.groupby('TableName', as_index=False)['LastDay'].sum()

# --- 4. Sortera tabeller efter lagring på sista dagen ---
df_sorted = df_grouped.sort_values(by='LastDay', ascending=False).reset_index(drop=True)

# --- 5. Beräkna total lagring och kumulativ andel ---
total_last_day_storage = df_sorted['LastDay'].sum()
df_sorted['CumulativeShare'] = df_sorted['LastDay'].cumsum() / total_last_day_storage

# --- 6. Hitta de tabeller som tillsammans står för 90% ---
df_top_90 = df_sorted[df_sorted['CumulativeShare'] <= 0.9]

# --- 7. Spara till CSV ---
output_path = r"C:\Users\b-elladjarv\Desktop\NetworkGraph\network_Structure_Retail.py\top_90_percent_lastday_tables.csv"
df_top_90[['TableName','LastDay','CumulativeShare']].to_csv(output_path, index=False)

# --- 8. Skriv ut för kontroll ---
print(f"\n✅ Klar! Fil sparad till: {output_path}")
print(f"Totalt antal tabeller: {len(df_sorted)}")
print("Tabeller som tillsammans motsvarar 90 % av total lagring (sista dagen):")
print(df_top_90[['TableName','LastDay','CumulativeShare']])
