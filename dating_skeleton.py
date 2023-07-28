import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("profiles.csv")

#print(df.body_type.head())
#print(df.income.head())
#print(df.diet.head())
#print(df.essay0.head())
#print(df.essay1.head())

#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16,80)
#plt.show()

#print(df.sign.value_counts())

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

smokes_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["smokes_code"] = df.smokes.map(smokes_mapping)

drugs_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drugs_code"] = df.drugs.map(drugs_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
 
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

all_essays = df[essay_cols].apply(lambda x: ' '.join(x), axis=1)
 
df["essay_len"] = df.apply(lambda x: len(x))

feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]
 
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
 
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

