# Machine Learning Classification Study
Estudo em Python para treinar Árvores de Decisão, Random Forest e Rede Neural Multilayer Perceptron.

### Sobre o código

1. consumer.py: classe para obter o csv com o dataset. O dataset escolhido foi o Wine Quality Dataset, que contém informações sobre características físicas e químicas de vinhos, e a classificação de sua qualidade. Esse dataset foi escolhido pois eu gosto de vinho. 
   
2. train_test_split.py: K-Fold Cross-Validation escolhido como abordagem de divisão do dataset em treinamento e teste. Foi escolhido pois reduz a variação nos resultados já que testa o modelo em várias divisões diferentes de dados.

3. decision_tree_classifier.py: treinamento de Árvore de Decisão

4. random_forest_classifier.py: treinamento de Random Forest

### Como executar o código?

1. Prepare o ambiente
```
python -m venv venv

venv/Scripts/activate

pip install -r requirements.txt
```

2. Escolha o arquivo e execute