from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

df = pd.read_csv("WineQT.csv", sep=",", encoding="utf-8")

df_quality = df['quality']
df = df.drop('quality', axis=1)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    df_quality_train, df_quality_test = df_quality.iloc[train_index], df_quality.iloc[test_index]
    
    # Cria o modelo
    model = RandomForestClassifier(random_state=42)

    # Treina
    model.fit(df_train, df_quality_train)

    # Predição
    pred_test = model.predict(df_test)

    # Avaliação
    accuracy = accuracy_score(df_quality_test, pred_test)
    f1 = f1_score(df_quality_test, pred_test, average='weighted')

    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 50)
