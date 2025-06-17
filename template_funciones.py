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











# ----------------------------------- FUNCIONES PARA LA SEGUNDA PARTE DEL TRABAJO -----------------------------------












def calcula_K(A):
   n, m = A.shape
   K = np.eye(n)
   for i in range(n):
     K[i,i] = sum(A[i, :])
   return K


def calcula_L(A):
  """
  Dada una matriz A calcula la matriz L = K - A.

  Parámetros
  ----------
    * A: matriz A simétrica.

  Devuelve
  ----------
    * L: matriz L.
  """
  K = calcula_K(A)
  L = K - A
  return L


def calcula_P(A):
    """
    Dada una matriz A calcula la matriz P, siendo P:

    P_ij = (K_ii * K_jj) / 2E

    Parámetros
    ----------
      * A: matriz A simétrica.

    Devuelve
    ----------
      * P: matriz P.
    """
    n, m = A.shape
    K = calcula_K(A)
    dos_E = np.sum(A)
    P = np.ones((n, m))
    for i in range(n):
      for j in range(m):
         P[i, j] = (K[i, i] * K[j, j]) / dos_E
    return P


def calcula_R(A):
    """
    Dada una matriz A calcula la matriz R, siendo R = A - P .

    Parámetros
    ----------
      * A: matriz A simétrica.

    Devuelve
    ----------
      * R: matriz R.
    """
    P = calcula_P(A)
    R = A - P
    return R


def calcula_lambda(L,v):
    """
    Dada una matriz A calcula la matriz Λ, siendo Λ

    Λ = (1/4) * (<s, <L, s>>)

    siendo <,> el producto interno usual y s el vector con los signos asociados
    a v.

    Parámetros
    ----------
      * L: matriz L.
      * v: autovector asociado a la matriz L.

    Devuelve
    ----------
      * Λ: matriz Λ.
    """
    s = np.sign(v)
    lambdon = (1/4) * (s @ L @ s)
    return lambdon


def calcula_Q(R,v):
    """
    Dada una matriz A calcula la matriz Q, siendo Q

    Q = (1/4E) * (<s, <R, s>>)

    siendo <,> el producto interno usual.

    Parámetros
    ----------
      * R: matriz R.
      * v: autovector asociado a la matriz R.

    Devuelve
    ----------
      * Q: matriz Q.
    """
    dos_E = np.sum(A)
    s = np.sign(v)
    Q = (1/(dos_E*2)) * (s @ R @ s)
    return Q


def metpot1(A,tol=1e-8,maxrep=np.inf):
  """
  Dada una matriz A calcula su autovalor de mayor módulo,
  con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones.

  Parámetros
  ----------
    * A: matriz A simétrica.
    * (tol): tolerancia de convergencia para el método.
    * (maxrep): cantidad máxima de iteraciones del método.

  Devuelve
  ----------
    * v1: autovector asociado al primer autovalor de A.
    * l1: primer autovalor de A.
    * bool: 'True' si el método llegó a converger.
  """
  v = np.random.rand(A.shape[1]) # Generamos un vector de partida aleatorio, entre -1 y 1
  v = v / np.linalg.norm(v) # Lo normalizamos
  v1 = A @ v # Aplicamos la matriz una vez
  v1 = v1 / np.linalg.norm(v1) # normalizamos
  l = v @ (A @ v) # Calculamos el autovector estimado
  l1 = v1 @ (A @ v1) # Y el estimado en el siguiente paso
  nrep = 0 # Contador
  while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada
     v = v1 # actualizamos v y repetimos
     l = l1
     v1 = A @ v # Calculo nuevo v1
     v1 = v1 / np.linalg.norm(v1) # Normalizo
     l1 = v1 @ (A @ v1) # Calculo autovector
     nrep += 1 # Un pasito mas
     if not nrep < maxrep:
       print('MaxRep alcanzado')
  l = v1 @ (A @ v1) # Calculamos el autovalor
  return v1,l,nrep<maxrep


def deflaciona(A,tol=1e-8,maxrep=np.inf):
  """
  Dada una matriz A devuelve una matriz A* sin el autovector
  asociado al autovalor de mayor valor en módulo.

  Parámetros
  ----------
    * A: matriz A simétrica.
    * (tol): tolerancia de convergencia para el método de la potencia.
    * (maxrep): cantidad máxima de iteraciones del método de la potencia.

  Devuelve
  ----------
    * deflA: matriz A* sin el autovector asociado al autovalor de mayor valor
    en módulo.
  """
  v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
  deflA = A - l1 * np.outer(v1, v1) / np.dot(v1, v1) # Sugerencia, usar la funcion outer de numpy
  return deflA


def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   """
   Dada una matriz A y calcula su segundo autovector de mayor módulo,
   con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones.

   Parámetros
   ----------
     * A: matriz A simétrica.
     * (tol): tolerancia de convergencia para el método de la potencia.
     * (maxrep): cantidad máxima de iteraciones del método de la potencia.

   Devuelve
   ----------
     * deflA: matriz A* sin el autovector asociado al primer autovalor.
   """
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   deflA = deflaciona(A)
   return metpot1(deflA,tol,maxrep)


def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
  """
  Retorna el primer autovalor de la inversa de A + mu * I,
  junto a su autovector y si el método convergió.

  Parámetros
  ----------
    * A: matriz A simétrica.
    * mu: valor para confeccionar la nueva matriz sobre la cual aplicar el método.
    * (tol): tolerancia de convergencia para el método de la potencia.
    * (maxrep): cantidad máxima de iteraciones del método de la potencia.

  Devuelve
  ----------
   * v1: autovector asociado al primer autovalor de A.
   * l1: primer autovalor de A.
   * bool: 'True' si el método llegó a converger.
  """
  A_shift = A + mu * np.eye(A.shape[0])
  return metpot1(inv(A_shift),tol=tol,maxrep=maxrep)


def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   """
   Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector
   de la matriz A, suponiendo que sus autovalores son positivos excepto por el
   menor que es igual a 0

   Parámetros
   ----------
     * A: matriz A simétrica.
     * mu: valor para confeccionar la nueva matriz sobre la cual aplicar el método.
     * (tol): tolerancia de convergencia para el método de la potencia.
     * (maxrep): cantidad máxima de iteraciones del método de la potencia.

   Devuelve
   ----------
   * v: autovector asociado al segundo autovalor de A.
   * l: segundo autovalor de A.
   * bool: 'True' si el método llegó a converger.
   """
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu * np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = inv(X) # La invertimos
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ = metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_
   

def recortar_grafo(A, l):
  """
  Parámetros
    ----------
    A: matriz A
    l: autovector.

  Devuelve
    ----------
    Ap: sub-matriz compuesta por nodos cuyo valor es positivo.
    An: sub-matriz compuesta por nodos cuyo valor es negativo.
  """
  pos_idx = []
  neg_idx = []
  for i in range(len(l)):
    if l[i] >= 0:
      pos_idx.append(i)
    elif l[i] < 0:
      neg_idx.append(i)

  Ap = np.zeros((len(pos_idx), len(pos_idx)))
  An = np.zeros((len(neg_idx), len(neg_idx)))

  for i in range(len(A)):
    for j in range(len(A)):
      if A[i][j] != 0:
        if i in pos_idx and j in pos_idx:
          ii = pos_idx.index(i)
          jj = pos_idx.index(j)
          Ap[ii][jj] = A[i][j]
        elif i in neg_idx and j in neg_idx:
          ii = neg_idx.index(i)
          jj = neg_idx.index(j)
          An[ii][jj] = A[i][j]

  return Ap, An


def laplaciano_iterativo(A,niveles,nombres_s=None):
    """
    Parámetros
      ----------
      A: matriz A
      l: autovector.

    Devuelve
      ----------
      Ap: sub-matriz compuesta por nodos cuyo valor es positivo.
      An: sub-matriz compuesta por nodos cuyo valor es negativo.
    """
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        deflL = deflaciona(L)
        v,l,_ = metpot1(deflL) # Encontramos el segundo autovector de L

        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap, Am = recortar_grafo(A, v) # Asociado al signo positivo, asociado al signo negativo

        return(
                # TODO: revisar
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>=0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    """
    Parámetros
      ----------
      A: matriz A
      R: matriz de modularidad
      nombres_s: nombres de los nodos

    Devuelve
      ----------
      Ap: sub-matriz compuesta por nodos cuyo valor es positivo.
      An: sub-matriz compuesta por nodos cuyo valor es negativo.
    """
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp, Rm = recortar_grafo(R,v) # Parte de R asociada a los valores positivos de v, parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                        modularidad_iterativo(A, Rp,
                                            nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                        modularidad_iterativo(A, Rm,
                                            nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                        )