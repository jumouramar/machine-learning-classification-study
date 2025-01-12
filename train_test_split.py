from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WineQT.csv", sep=",", encoding="utf-8")

# Separando o dataframe entre demais dados e resultado
df_quality = df['quality']
df = df.drop('quality', axis=1)

# Ajustar a escala dos atributos para que todos tenham o mesmo intervalo.
# Alguns algoritmos de aprendizado de máquina funcionam melhor quando 
# todas as variáveis têm uma escala similar.
# df = StandardScaler().fit_transform(df)

# Configuração do KFold
kf = KFold(n_splits=4, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
    # Dividir os dados de treino e teste
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    df_quality_train, df_quality_test = df_quality.iloc[train_index], df_quality.iloc[test_index]
    
    print(f"Fold {fold}:")
    print(f"Dados de treino: {df_train.shape[0]}")
    print(f"Dados de teste: {df_test.shape[0]}")
    print("-" * 50)
