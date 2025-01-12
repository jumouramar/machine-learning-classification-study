import pandas as pd

# Carregar o dataframe, adicionar uma identificação, 
# filtrar pelos melhores parâmetros e filtrar as colunas.


# DECISION TREE
df_decision_tree = pd.read_excel('results/decision_tree.xlsx')
df_decision_tree["algoritmo"] = "Decision Tree"
df_decision_tree = df_decision_tree.drop(columns=["fold"])

df_decision_tree_result = df_decision_tree.groupby(
    ['algoritmo', 'criterion', 'max_depth']).agg(
    {'accuracy': 'mean', 'f1_score': 'mean'}
).reset_index()

best_result_row = df_decision_tree_result.loc[df_decision_tree_result['accuracy'].idxmax()]
criterion_best_result = best_result_row['criterion']
max_depth_best_result = best_result_row['max_depth']

df_decision_tree = df_decision_tree[
    (df_decision_tree["criterion"] == criterion_best_result) & 
    (df_decision_tree["max_depth"] == max_depth_best_result)
]

df_decision_tree = df_decision_tree[["algoritmo", "accuracy", "f1_score"]]

print(df_decision_tree)


# RANDOM FOREST
df_random_forest = pd.read_excel('results/random_forest.xlsx')
df_random_forest["algoritmo"] = "Random Forest"
df_random_forest = df_random_forest.drop(columns=["fold"])

df_random_forest_result = df_random_forest.groupby(
    ['algoritmo', 'n_estimators', 'max_depth']).agg(
    {'accuracy': 'mean', 'f1_score': 'mean'}
).reset_index()

best_result_row = df_random_forest_result.loc[df_random_forest_result['accuracy'].idxmax()]
n_estimators_best_result = best_result_row['n_estimators']
max_depth_best_result = best_result_row['max_depth']

df_random_forest = df_random_forest[
    (df_random_forest["n_estimators"] == n_estimators_best_result) & 
    (df_random_forest["max_depth"] == max_depth_best_result)
]

df_random_forest = df_random_forest[["algoritmo", "accuracy", "f1_score"]]

print(df_random_forest)


# MLP
df_mlp = pd.read_excel('results/mlp.xlsx')
df_mlp["algoritmo"] = "MLP"
df_mlp = df_mlp.drop(columns=["fold"])

df_mlp_result = df_mlp.groupby(
    ['algoritmo', 'hidden_layer_sizes', 'max_iter']).agg(
    {'accuracy': 'mean', 'f1_score': 'mean'}
).reset_index()

best_result_row = df_mlp_result.loc[df_mlp_result['accuracy'].idxmax()]
hidden_layer_sizes_best_result = best_result_row['hidden_layer_sizes']
max_iter_best_result = best_result_row['max_iter']

df_mlp = df_mlp[
    (df_mlp["hidden_layer_sizes"] == hidden_layer_sizes_best_result) & 
    (df_mlp["max_iter"] == max_iter_best_result)
]

df_mlp = df_mlp[["algoritmo", "accuracy", "f1_score"]]

print(df_mlp)


# FINAL
# Concatenar os dataframes e encontrar a melhor média
df = pd.concat([df_decision_tree, df_random_forest, df_mlp], ignore_index=True)

df_result = df.groupby('algoritmo').agg({'accuracy': 'mean', 'f1_score': 'mean'})

print(df_result)
