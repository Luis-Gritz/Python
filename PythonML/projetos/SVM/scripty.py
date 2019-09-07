import numpy as np
from sklearn import svm
#carrega a base
x = np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y = np.array([1,1,2,2])

#cria o classificador
clf = svm.SVC(gamma='auto')
#print(type(clf))

#treina o classificador
clf.fit(x,y)

print(clf.support_vectors_)
print(clf.n_support_)

v1 = [-2,-2]
v2 = [2,2]

v= [v1,v2]


print(clf.predict([v1]))

print(clf.predict([v2]))

print(clf.predict(v))
