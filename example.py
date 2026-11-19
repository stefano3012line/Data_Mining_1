import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import OPTICS
import numpy as np
import sklearn.preprocessing
#sns.pairplot(df,palette="husl",kind="hist",hue=df.columns[0],diag_kind="kde",corner=True)
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt=".3f",annot_kws={"size": 6}).tick_params(axis='both', which='major', labelsize=6)#matrice delle correlazioni
#sns.heatmap(df.cov(), annot=True, cmap='coolwarm',annot_kws={"size": 4}).tick_params(axis='both', which='major', labelsize=6)#matrice di covarianza
#print(df.info())# Non-Null Count + Dtype 
#sns.countplot(data=df)


'''
#count+mean+std+min+25%+50%+75%+max
with open("count+mean+std+min+25+50+75%+max.txt", "w") as f: 
    f.write(df.describe().to_string())
'''

#print(df.duplicated(df.columns[0])) #conta i duplicati

'''
#conta i valori unici, nulli e duplicati per ogni colonna
with open("unique+null+duplicated.txt", "w") as f:
    for col in df.columns:
        f.write(f"Colonna: {col}\n")
        f.write(f"Valori unici: {df[col].nunique()}\n")
        f.write(f"Valori nulli: {df[col].isnull().sum()}\n")
        f.write(f"Valori duplicati: {df.duplicated(subset=col).sum()}\n")
        f.write("\n")
    print("\n")
#df.isnull().sum() #conta i valori nulli per ogni colonna
'''



#histograms for each column
'''
for col in df.columns:
    plt.figure(figsize=(10,6))
    df[col].hist(log=False)
    plt.title(col)
    plt.savefig("histograms/"f"{col}_histogram.png")
'''   

'''
scaler=sklearn.preprocessing.StandardScaler()
df_scaled=scaler.fit_transform(df)


#histograms for each column

for col in df_scaled.columns:
    plt.figure(figsize=(10,6))
    df_scaled[col].hist(log=True)
    plt.title(col)
    plt.show()
    #plt.savefig("histograms_logaritmic/"f"{col}_histogram_logaritmic.png")   
'''

#sns.heatmap(data=df_scaled.corr(), annot=True, cmap='coolwarm')
#sns.clustermap(data=df.corr(),annot=True)
#sns.dogplot(data=df)  #Good Boy
#sns.catplot(data=df, x=df.columns[0][::50], y=df.columns[1][::50], hue=df.columns[0], kind='boxen')
#sns.catplot(data=df, kind='boxen',y=df.columns[1])
#boxplot di tuttos
'''
for col in df.columns:
    plt.figure(figsize=(10,6))
    sns.countplot(data=df[col])
    plt.title(col)
    plt.savefig("countplot/"f"{col}_boxplot.png") 
'''
#sns.kdeplot(data=df, x=df.columns[0], y=df.columns[-1], hue=df.columns[0], fill=True, alpha=0.5)
#clustering=OPTICS(min_samples=4).fit_predict(df[[df.columns[2], df.columns[3]]])
#print(clustering)
#sns.scatterplot(data=df, x=df.columns[0], y=df.columns[-1], hue=df.columns[0], style=df.columns[0], s=100)
#plt.tight_layout()
alim=12
fv= 2.2
'''target_i=0.1
r= (alim-4*fv)/target_i
power= r*(target_i**2)
print(r," Ohm")
print(power," W")'''


r=10

current= (alim-5*fv)/r

power_r= r*(current**2)
power_i = alim*current*3
print(current," A")
print(power_r," W")
print(power_i," W")