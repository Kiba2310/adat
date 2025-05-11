import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adatok beolvasása
df = pd.read_csv('winemag-data-130k-v2.csv')

# Szűrjük ki a hiányzó árakat tartalmazó sorokat és készítsünk egy másolatot
df_clean = df.dropna(subset=['price']).copy()  # .copy() hozzáadva

# Átlagos árak országonként
average_prices = df_clean.groupby('country')['price'].mean().sort_values(ascending=False)
print("\nTop 10 countries with highest average wine prices:")
print(average_prices.head(10))

# Részletes árstatisztikák országonként
price_stats = df_clean.groupby('country').agg({
    'price': ['count', 'mean', 'median', 'std', 'min', 'max']
}).sort_values(('price', 'mean'), ascending=False)

print("\nRészletes árstatisztikák a top 10 országra:")
print(price_stats.head(10))

# Átlagos pontszám és ár összehasonlítása országonként
country_stats = df_clean.groupby('country').agg({
    'price': 'mean',
    'points': 'mean'
}).sort_values('price', ascending=False)

print("\nÁtlagos ár és pontszám országonként (top 10):")
print(country_stats.head(10))

# Ár kategóriák létrehozása és elemzése
df_clean.loc[:, 'price_category'] = pd.qcut(df_clean['price'], q=5, labels=['Nagyon olcsó', 'Olcsó', 'Közepes', 'Drága', 'Nagyon drága'])
price_points = df_clean.groupby('price_category', observed=True).agg({
    'points': 'mean',
    'price': ['count', 'mean']
})

print("\nÁrkategóriánkénti statisztikák:")
print(price_points)

# Korreláció minden numerikus oszlop között
numeric_cols = df_clean[['price', 'points']]
correlation_matrix = numeric_cols.corr()
print("\nKorrelációs mátrix:")
print(correlation_matrix)

# Extra statisztikák
print("\nÁltalános statisztikák:")
print(f"Átlagos borár: {df_clean['price'].mean():.2f}")
print(f"Medián borár: {df_clean['price'].median():.2f}")
print(f"Legdrágább bor ára: {df_clean['price'].max():.2f}")
print(f"Legolcsóbb bor ára: {df_clean['price'].min():.2f}")
print(f"Árak szórása: {df_clean['price'].std():.2f}")

# Árak percentilisei
valid_prices = df_clean['price'].values
percentiles = [10, 25, 50, 75, 90, 95, 99]
price_percentiles = np.percentile(valid_prices, percentiles)
print("\nÁr percentilisek:")
for p, v in zip(percentiles, price_percentiles):
    print(f"{p}. percentilis: {v:.2f}")

# Vizualizáció
plt.figure(figsize=(12, 6))
sns.boxplot(x='country', y='price', data=df_clean)
plt.xticks(rotation=90)
plt.title('Bor árak országonként')
plt.tight_layout()
plt.savefig('bor_arak_orszagonkent.png')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Magyar karakterek megfelelő megjelenítése
plt.rcParams['font.family'] = 'DejaVu Sans'

# Adatok beolvasása
df = pd.read_csv('winemag-data-130k-v2.csv')
df_clean = df.dropna(subset=['price']).copy()

# 1. Ábra: Top 10 ország átlagárai
plt.figure(figsize=(12, 6))
average_prices = df_clean.groupby('country')['price'].mean().sort_values(ascending=False)
average_prices.head(10).plot(kind='bar')
plt.title('Top 10 ország átlagos borárai')
plt.xlabel('Ország')
plt.ylabel('Átlagár ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top10_orszag_atlagarak.png')
plt.close()

# 2. Ábra: Ár és pontszám összefüggése
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='points', y='price', alpha=0.5)
plt.title('Borok ára és pontszáma közötti összefüggés')
plt.xlabel('Pontszám')
plt.ylabel('Ár ($)')
plt.tight_layout()
plt.savefig('ar_pontszam_osszefugges.png')
plt.close()

# 3. Ábra: Árkategóriák létrehozása és vizualizációja
df_clean['price_category'] = pd.qcut(df_clean['price'],
                                   q=5,
                                   labels=['Nagyon olcsó', 'Olcsó', 'Közepes', 'Drága', 'Nagyon drága'])

plt.figure(figsize=(10, 6))
df_clean['price_category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Borok eloszlása árkategóriák szerint')
plt.tight_layout()
plt.savefig('arkategoriak_eloszlasa.png')
plt.close()

# 4. Ábra: Pontszámok árkategóriánként
plt.figure(figsize=(10, 6))
sns.boxplot(x='price_category', y='points', data=df_clean)
plt.title('Pontszámok eloszlása árkategóriánként')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pontszamok_arkategoriankent.png')
plt.close()

# 5. Ábra: Árak eloszlása országonként
plt.figure(figsize=(12, 6))
top_10_countries = average_prices.head(10).index
sns.violinplot(x='country', y='price',
               data=df_clean[df_clean['country'].isin(top_10_countries)])
plt.title('Árak eloszlása a top 10 országban')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('arak_eloszlasa_orszagonkent.png')
plt.close()

# 6. Ábra: Korrelációs heatmap
correlation_matrix = df_clean[['price', 'points']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Korrelációs heatmap')
plt.tight_layout()
plt.savefig('korrelacios_heatmap.png')
plt.close()

# 7. Ábra: Árak eloszlása (hisztogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=df_clean[df_clean['price'] <= df_clean['price'].quantile(0.95)],
             x='price', bins=50)
plt.title('Borárak eloszlása (95. percentilisig)')
plt.xlabel('Ár ($)')
plt.ylabel('Gyakoriság')
plt.tight_layout()
plt.savefig('arak_hisztogram.png')
plt.close()

# 8. Ábra: Szőlőfajták átlagárai
plt.figure(figsize=(12, 6))
variety_prices = df_clean.groupby('variety')['price'].mean().sort_values(ascending=False).head(10)
variety_prices.plot(kind='bar')
plt.title('Top 10 legdrágább szőlőfajta átlagárai')
plt.xlabel('Szőlőfajta')
plt.ylabel('Átlagár ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('szolofajtak_atlagarak.png')
plt.close()

print("A következő vizualizációk készültek el:")
print("1. top10_orszag_atlagarak.png - Top 10 ország átlagárai")
print("2. ar_pontszam_osszefugges.png - Ár és pontszám összefüggése")
print("3. arkategoriak_eloszlasa.png - Árkategóriák eloszlása")
print("4. pontszamok_arkategoriankent.png - Pontszámok árkategóriánként")
print("5. arak_eloszlasa_orszagonkent.png - Árak eloszlása országonként")
print("6. korrelacios_heatmap.png - Korrelációs heatmap")
print("7. arak_hisztogram.png - Árak eloszlása")
print("8. szolofajtak_atlagarak.png - Szőlőfajták átlagárai")