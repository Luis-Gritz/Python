import numpy as np
from sklearn import svm, datasets, metrics

#iris = datasets.load_iris()
#iris = datasets.load_digits()
iris = datasets.load_breast_cancer



x = iris.data
y = iris.target

#print(x)
#print(y)

np.random.seed(0)

n_amostras = len(x)

ordem = np.random.permutation(n_amostras)

porcentagem = 0.7

x = x[ordem]
y = y[ordem]

x_treino = x[:int(porcentagem*n_amostras)]
y_treino = y[:int(porcentagem*n_amostras)]

x_teste = x[int(porcentagem*n_amostras):]
y_teste = y[int(porcentagem*n_amostras):]

#clf = svm.SVC(gamma='auto')
clf = svm.SVC(gamma='scale')

clf.fit(x_treino,y_treino)

print(clf.support_vectors_)
print(clf.n_support_)


predicao = clf.predict(x_teste)

print(predicao)

taxa_acerto = clf.score(x_teste,y_teste)
print(taxa_acerto)

matriz = metrics.confusion_matrix(y_teste,predicao)

for item in matriz:
    print(item)