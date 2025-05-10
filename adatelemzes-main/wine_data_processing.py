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