import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

df = pd.read_csv("WineQT.csv", sep=",", encoding="utf-8")

df_quality = df['quality']
df = df.drop('quality', axis=1)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

df_decision_tree = pd.DataFrame(columns=['criterion', 'max_depth', 'fold', 'accuracy', 'f1_score'])

# Parâmetros
max_depth_values = [None, 1, 5, 10, 25]
criterion_value = ["gini", "entropy"]

for criterion in criterion_value:
    for max_depth in max_depth_values:
        for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]
            df_quality_train, df_quality_test = df_quality.iloc[train_index], df_quality.iloc[test_index]
            
            # Cria o modelo
            model = DecisionTreeClassifier(
                criterion=criterion, 
                max_depth=max_depth, 
                random_state=42
            )

            # Treina
            model.fit(df_train, df_quality_train)

            # Predição
            pred_test = model.predict(df_test)

            # Avaliação
            accuracy = accuracy_score(df_quality_test, pred_test)
            f1 = f1_score(df_quality_test, pred_test, average='weighted')

            print(f"Criterion: {criterion}")
            print(f"Max Depth: {max_depth}")
            print(f"Acurácia: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 50)

            df_decision_tree.loc[len(df_decision_tree)] = [criterion, max_depth, fold, accuracy, f1]

df_decision_tree.to_excel('results/decision_tree.xlsx', index=False)
