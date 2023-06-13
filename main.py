import numpy as np

np.set_printoptions(precision=3)  # Ajusta a precisão de impressão do numpy para 3 casas decimais

def gauss_elimination(A, b):
    n = len(A)
    P = np.eye(n)  # Matriz de Permutação
    L = np.eye(n)  # Matriz Lower
    U = A.copy()  # Matriz Upper
    Ab = np.column_stack((A, b))

    for i in range(n):
        pivot_row = np.argmax(np.abs(U[i:, i])) + i

        # Troca de linhas
        P[[i, pivot_row]] = P[[pivot_row, i]]
        U[[i, pivot_row]] = U[[pivot_row, i]]
        L[[i, pivot_row], :i] = L[[pivot_row, i], :i]

        print(f'\nTrocando a linha {i} com a linha de pivot {pivot_row} se necessário:')
        print(U)

        # Divisão pela linha pivô
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]

            print(f'\nSubtraindo a linha {i} da linha {j}:')
            print(U)

    print('\n______________LU_____________')  # Adicionado aqui
    print('\nMatriz L:')
    print(L)
    print('\nMatriz U:')
    print(U)
    print('\nMatriz P:')
    print(P)

    # Resolução Ly=Pb
    Pb = np.dot(P, b)
    y = np.linalg.solve(L, Pb)

    # Resolução Ux=y
    x = np.linalg.solve(U, y)

    print("\nSolução:")
    print(x)

    return x

# Sistema
A = np.array([[1, 6, 2, 4],
              [3, 19, 4, 15],
              [1, 4, 8, -12],
              [5, 33, 9, 3]], dtype=float)  # Assegurando que A seja uma matriz de floats

b = np.array([8, 25, 18, 72], dtype=float)  # Assegurando que b seja um vetor de floats

# Execução do Gauss
solution = gauss_elimination(A, b)
