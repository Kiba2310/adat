from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud

# Seaborn téma beállítása
sns.set_theme()

# Az adatok betöltése és feldolgozása
# (itt folytatódik az eredeti kód többi része)
# Stílus beállítások a vizualizációkhoz
sns.set_palette("husl")


# Adatok beolvasása
df = pd.read_csv('winemag-data-130k-v2.csv')
df_clean = df.dropna(subset=['price']).copy()

# === 1. ALAPSTATISZTIKÁK ÉS ELOSZLÁSOK ===
print("\n=== Alapstatisztikák ===")
print(df_clean.describe())

# Hiányzó értékek vizsgálata
missing_data = df_clean.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_data / len(df_clean) * 100).round(2)
print("\n=== Hiányzó értékek ===")
print(pd.DataFrame({'Hiányzó értékek': missing_data, 'Százalék': missing_percent}))

# === 2. ÁRAK RÉSZLETES ELEMZÉSE ===
price_stats = df_clean.groupby('country').agg({
    'price': ['count', 'mean', 'median', 'std', 'min', 'max']
}).sort_values(('price', 'count'), ascending=False)
price_stats = price_stats[price_stats[('price', 'count')] >= 100]  # Csak jelentős mintaszámú országok

print("\n=== Top 10 ország borstatisztikái ===")
print(price_stats.head(10))

# Ár-érték arány számítása
df_clean['price_per_point'] = df_clean['price'] / df_clean['points']
best_value = df_clean.nsmallest(10, 'price_per_point')[
    ['title', 'country', 'points', 'price', 'price_per_point']]
print("\n=== Legjobb ár-érték arányú borok ===")
print(best_value)

# === 3. VIZUALIZÁCIÓK ===

# Ár eloszlás
plt.figure(figsize=(12, 6))
sns.histplot(data=df_clean[df_clean['price'] <= df_clean['price'].quantile(0.95)],
            x='price', bins=50)
plt.title('Borárak eloszlása (95. percentilisig)')
plt.savefig('ar_eloszlas.png')
plt.close()

# Pontszám vs. Ár
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='points', y='price', alpha=0.5)
plt.title('Pontszám és Ár összefüggése')
plt.savefig('pontszam_ar_kapcsolat.png')
plt.close()

# === 4. SZÖVEGES ELEMZÉS ===

# Leggyakoribb szavak a leírásokban
def get_word_frequencies(text_series):
    all_words = ' '.join(text_series.dropna()).lower().split()
    return Counter(all_words).most_common(20)

print("\n=== Leggyakoribb szavak a leírásokban ===")
word_freq = get_word_frequencies(df_clean['description'])
print(word_freq)

# Szófelhő készítése
text = ' '.join(df_clean['description'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('bor_leirasok_szofelho.png')
plt.close()

# === 5. KATEGÓRIA ELEMZÉS ===

# Szőlőfajták elemzése
variety_stats = df_clean.groupby('variety').agg({
    'price': ['count', 'mean'],
    'points': 'mean'
}).sort_values(('price', 'count'), ascending=False)
variety_stats = variety_stats[variety_stats[('price', 'count')] >= 100]

print("\n=== Top 10 szőlőfajta statisztikái ===")
print(variety_stats.head(10))

# === 6. IDŐBELI ELEMZÉS (ha van évjárat adat) ===
if 'vintage' in df_clean.columns:
    vintage_stats = df_clean.groupby('vintage').agg({
        'price': ['mean', 'count'],
        'points': 'mean'
    }).sort_index()

    plt.figure(figsize=(15, 6))
    vintage_stats[('price', 'mean')].plot()
    plt.title('Átlagos borárak évjárat szerint')
    plt.savefig('arak_evjarat_szerint.png')
    plt.close()

# === 7. KORRELÁCIÓS ELEMZÉS ===
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
correlation_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korrelációs mátrix')
plt.savefig('korrelacios_matrix.png')
plt.close()

# === 8. RÉSZLETES STATISZTIKÁK EXPORTÁLÁSA ===
# Excel helyett CSV formátumban mentjük az adatokat
price_stats.to_csv('orszag_statisztikak.csv')
variety_stats.to_csv('szolofajtak.csv')
best_value.to_csv('legjobb_arertek.csv')

min_price = df_clean['price'].min()
print(f"Minimális ár: {min_price}")