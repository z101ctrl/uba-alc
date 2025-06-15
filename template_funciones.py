import numpy as np
import scipy

def construye_adyacencia(D,m):
    """
    Construye la matriz de adyacencia del grafo de museos
 
    Input:
       D: matriz de distancias
       m: Cantidad de links por nodo
     
    Output:
       Matriz de adyacencia como un array de numpy
    """
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)


def calcularLU(matriz):
    """
    Obtiene la factorizacion LU de una matriz
 
    Input:
       matriz: matriz de nxn
     
    Output:
       Tupla (L,U)
    """
    m, n = matriz.shape
    if m != n:
        raise ValueError('La matriz debe ser cuadrada')

    U = matriz.astype(float).copy()
    L = np.eye(n)

    # Recorremos las columnas
    for c in range(n):
        # Elemento de la diagonal
        pivote = U[c, c]
        # Recorremos las filas a partir del indice c+1 
        for f in range(c + 1, n):
            escalar = U[f, c] / pivote
            # Guardamos el escalar utilizado
            L[f, c] = escalar
            # Obtenemos un 0 en U[f, c]
            U[f] = U[f] - escalar * U[c]

    return L, U


def calcula_Kinv(A):
    """
    Obtiene la matriz K inv a partir de A.
 
    Input:
       A: matriz de nxn
     
    Output:
       Matriz K invertida
    """
    n, m = A.shape
    K_inv = np.eye(n)
    for i in range(n):
      # Obs: En caso de que la suma de una columna sea 0 se coloca un 0 (para evitar 1/0)
      K_inv[i,i] = 1/sum(A[i,:]) if sum(A[i,:]) != 0 else 0
    return K_inv


def calcula_matriz_C(A):
    """
    Obtiene la matriz C
 
    Input:
       A: matriz de nxn
     
    Output:
       Matriz C
    """
    K_inv = calcula_Kinv(A)
    C = A.T @ K_inv
    return C


def calcular_inv_LU(L, U):
    """
    Calcula inversa de una matriz A dada su factorizacion LU.

    Input:
      L: matriz triangular inferior
      U: matriz triangular superior

    Output:
      Inversa de A
    """
    I = np.eye(U.shape[0])
    Y = scipy.linalg.solve_triangular(L, I, lower=True)
    X = scipy.linalg.solve_triangular(U, Y)

    return X

    
def calcula_pagerank(A, alfa):
    """
    Calcula PageRank de A

    Input:
      A: matriz de nxn
      alfa: coeficiente de damping

    Output:
      Vector p con los coeficientes de page rank de cada museo
    """
    C = calcula_matriz_C(A)
    N = A.shape[0]
    M = (N/alfa) * (np.identity(N) - (1 - alfa) * C)
    L, U = calcularLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.array([1] * N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    invM = calcular_inv_LU(L, U)
    return invM @ b


def calcula_matriz_C_continua(D):
    """
    Función para calcular la matriz de transiciones C

    Input:
     A: Matriz de adyacencia

    Output:
      Matriz C en versión continua
    """
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)

    # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    Kinv = calcula_Kinv(F)
    # Calcula C multiplicando Kinv y F
    C = F @ Kinv

    return C


def norma_1_matricial(A):
    """
    Devuelve norma 1 matricial.
  
    Input:
      A: matriz de NxN inversible
  
    Output:
      norma 1 de A
    """
    return np.max(np.sum(np.abs(A), axis=0))


def inv(A):
    """
    Devuelve inversa de matriz A.

    Input:
      A: matriz de NxN

    Output:
      Inversa de A
    """
    A = A.copy()
    L, U = calcularLU(A)
    B = calcular_inv_LU(L, U)
    return B


def numero_de_condicion_1(A):
    """
    Devuelve el numero de condicion 1 de la matriz A

    Input:
      A: matriz de nxn

    Output:
      Numero de condicion 1 de A
    """
    return norma_1_matricial(A) * norma_1_matricial(inv(A))


def calcula_B(C,r):
    """
    Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    suponiendo que cada visitante realizó cantidad_de_visitas pasos.
  
    Input:
      C: Matriz de transiciones
      cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
  
    Output:
      B: matriz que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    """
    # Inicializamos B con la matriz identidad
    B = np.eye(C.shape[0])
    for i in range(r-1):
      # Ultima potencia de C calculada
      power = np.identity(C.shape[0])
      for j in range(i+1):
        power = power @ C
      # Sumatoria de las potencias de C
      B = B + power
  
    return B