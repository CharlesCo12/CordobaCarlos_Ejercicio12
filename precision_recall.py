import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve

#Definimos el F1-Score
def F1(prec,cob):
    return 2*(prec*cob)/(prec+cob)

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0


# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#CALCULO DEL PCA SOBRE LOS 1

# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

linear.fit(x_test_transform[:,:10], y_test)
prob=linear.predict_proba(x_test_transform[:,:10])
pca_1=precision_recall_curve(y_test,prob[:,1])

#CALCULO DEL PCA SOBRE LOS 0

# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 0
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

linear.fit(x_test_transform[:,:10], y_test)
prob=linear.predict_proba(x_test_transform[:,:10])
pca_2=precision_recall_curve(y_test,prob[:,1])

#CALCULO DEL PCA SOBRE TODOS

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

linear.fit(x_test_transform[:,:10], y_test)
prob=linear.predict_proba(x_test_transform[:,:10])
pca_3=precision_recall_curve(y_test,prob[:,1])

#GRÁFICA DE LOS 3 FIT
i_1=np.where(F1(pca_1[0][:-1],pca_1[1][:-1])==np.max(F1(pca_1[0][:-1],pca_1[1][:-1])))
i_2=np.where(F1(pca_2[0][:-1],pca_2[1][:-1])==np.max(F1(pca_2[0][:-1],pca_2[1][:-1])))
i_3=np.where(F1(pca_3[0][:-1],pca_3[1][:-1])==np.max(F1(pca_3[0][:-1],pca_3[1][:-1])))
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot(pca_1[1],pca_1[0],label='PCA sobre 1')
plt.scatter(pca_1[1][i_1[0][0]],pca_1[0][i_1[0][0]],c='red')
plt.plot(pca_2[1],pca_2[0],label='PCA sobre 0')
plt.scatter(pca_2[1][i_2[0][0]],pca_2[0][i_2[0][0]],c='red')
plt.plot(pca_3[1],pca_3[0],label='PCA sobre Todos')
plt.scatter(pca_3[1][i_3[0][0]],pca_3[0][i_3[0][0]],c='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc=0.0)

plt.subplot(122)
plt.plot(pca_1[2],F1(pca_1[0][:-1],pca_1[1][:-1]),label='PCA sobre 1')
plt.scatter(pca_1[2][i_1[0][0]],F1(pca_1[0][:-1],pca_1[1][:-1])[i_1[0][0]],c='red')
plt.plot(pca_2[2],F1(pca_2[0][:-1],pca_2[1][:-1]),label='PCA sobre 0')
plt.scatter(pca_2[2][i_2[0][0]],F1(pca_2[0][:-1],pca_2[1][:-1])[i_2[0][0]],c='red')
plt.plot(pca_3[2],F1(pca_3[0][:-1],pca_3[1][:-1]),label='PCA sobre Todos')
plt.scatter(pca_3[2][i_3[0][0]],F1(pca_3[0][:-1],pca_3[1][:-1])[i_3[0][0]],c='red')
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.legend(loc=0.0)

plt.savefig('F1_prec_recall.png')
plt.close()