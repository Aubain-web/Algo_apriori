import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Charger le fichier CSV depuis le même dossier
df = pd.read_csv("data_transactions.csv")

# 2. Afficher les premières lignes
print("Premières lignes du fichier CSV :")
print(df.head())

# 3. Transformer les données en liste
dataset = df['ItemsAchetés'].apply(lambda x: x.split(','))  # Remplacer 'ItemsAchetés' par le nom de ta colonne

# 4. Convertir les données en format binaire
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

# 5. Créer un DataFrame binaire
df_bin = pd.DataFrame(te_ary, columns=te.columns_)

# Afficher le DataFrame binaire
print("Données binaires :")
print(df_bin.head())

# 6. Appliquer l'algorithme Apriori avec un support de 22%
items_frequents = apriori(df_bin, min_support=0.22, use_colnames=True)

# 7. Afficher les itemsets fréquents
print("Items fréquents :")
print(items_frequents)

# 8. Extraire les règles d'association avec une confiance minimale de 70%
regles_asso = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=2)

# 9. Afficher les règles d'association
print("\nRègles d'association :")
print(regles_asso)
