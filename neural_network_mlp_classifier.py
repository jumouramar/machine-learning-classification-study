import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

df = pd.read_csv("WineQT.csv", sep=",", encoding="utf-8")

df_quality = df['quality']
df = df.drop('quality', axis=1)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

df_mlp = pd.DataFrame(columns=['hidden_layer_sizes', 'max_iter', 'fold', 'accuracy', 'f1_score'])

# Parâmetros
hidden_layer_sizes_values = [(50,), (100,), (50, 50), (100, 50)]
max_iter_values = [100, 500, 1000]

for hidden_layer_sizes in hidden_layer_sizes_values:
    for max_iter in max_iter_values:
        for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]
            df_quality_train, df_quality_test = df_quality.iloc[train_index], df_quality.iloc[test_index]
            
            # Cria o modelo
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes, 
                max_iter=max_iter,
                random_state=42
            )

            # Treina
            model.fit(df_train, df_quality_train)

            # Predição
            pred_test = model.predict(df_test)

            # Avaliação
            accuracy = accuracy_score(df_quality_test, pred_test)
            f1 = f1_score(df_quality_test, pred_test, average='weighted')

            print(f"{hidden_layer_sizes}")
            print(f"{max_iter}")
            print(f"Acurácia: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 50)

            df_mlp.loc[len(df_mlp)] = [hidden_layer_sizes, max_iter, fold, accuracy, f1]

df_mlp.to_excel('results/mlp.xlsx', index=False)
