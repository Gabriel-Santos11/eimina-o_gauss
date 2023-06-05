import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)

    P = np.eye(n)
    for k in range(n - 1):
        # Pivoteamento parcial
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        P[[k, pivot_row]] = P[[pivot_row, k]]
        A[[k, pivot_row]] = A[[pivot_row, k]]

        # Atualizar L e U
        L[k+1:, k] = A[k+1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        U[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k:k+1] @ U[k:k+1, k+1:]

    return L, U, P

# Exemplo de matriz
A = np.array([[2, 3, -1],
              [4, -2, 3],
              [1, 1, 1]])

# Calcular a decomposição LU com permutação
L, U, P = lu_decomposition(A)

print("Matriz L:")
print(L)
print("Matriz U:")
print(U)
print("Matriz P:")
print(P)