import numpy as np
import operator as op

def perceptron(alfa,b,Muestras,C,iteraciones):
    Vectores = Muestras[0]
    Clases = Muestras[1]
    N = len(Vectores)
    D = len(Vectores[0]) # D = dimension del espacio vectorial
    a = [np.array([0.0 for j in range(D)]) for i in range(C)]
    m = 0
    it = 0
    while m<N:
        m = 0  # Aqui estaba el error, no habia puesto esta sentencia
        for n,y_n in enumerate(Vectores):
            i = Clases[n]
            g = np.transpose(a[i]).dot(y_n)
            error = False
            for j in range(C):
                if i!=j:
                    if np.transpose(a[j]).dot(y_n)+b > g:
                        a[j] -= alfa*y_n
                        error = True
            if error:
                a[i] += alfa*y_n
            else:
                m+=1
        it+=1
        if it==iteraciones:
            print("AVISO: Se alcanzo el limite de iteraciones")
            break
    return a

Vectores = []
Clases = []
C = {}
inversa_C = {}
# Entrenamiento #
with open("entrenamiento.txt") as fichero_muestras:
    for linea in fichero_muestras:
        l = linea.rstrip().split()
        parte_vector = map(float,l[0:-1])
        vector = np.array([1.0]+parte_vector)
        clase_string = l[-1]
        if clase_string not in C:
            C[clase_string] = len(C)
            inversa_C[len(C)-1] = clase_string
        clase = int(C[clase_string])
        Vectores.append(vector)
        Clases.append(clase)
print("Introduzca el limite de iteraciones: ")
iteraciones = int(input())
print("Introduzca el factor de aprendizaje alfa: ")
alfa = float(input())
print("Introduzca el margen b: ")
b = float(input())
a = perceptron(alfa,b,(Vectores,Clases),len(C),iteraciones)
# Clasificacion #
Muestras_clasif = []
with open("clasificacion.txt") as muestras:
    for linea in muestras:
        l = map(float,linea.rstrip().split())
        Muestras_clasif.append(np.array([1.0]+l))
resultado = []
for muestra in Muestras_clasif:
    puntuaciones = [(i,np.transpose(a_i).dot(muestra)) for i,a_i in enumerate(a)]
    arg_maximo = max(puntuaciones, key=op.itemgetter(1))[0]
    resultado.append(arg_maximo)
clasificadas = [[] for clase in C]
for i,clase_c in enumerate(resultado):
    m = Muestras_clasif[i][1:] # A coordenadas no homogéneas
    clasificadas[clase_c].append(str(m))
f_clasificadas = open("resultado.txt","w")
for i,muestras_clase in enumerate(clasificadas):
    str_muestras = " ".join(map(str,muestras_clase))
    f_clasificadas.write(inversa_C[i]+" "+str_muestras+"\n")
f_clasificadas.close()
