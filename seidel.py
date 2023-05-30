import numpy as np

def gauss_seidel(A, b, N, epsilon):
    n = len(A)
    x = np.zeros(n)

    for _ in range(N):
        x_new = np.zeros(n)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x) < epsilon:
            break

        x = x_new

    return x

# Exemplo de sistema de equações
A = np.array([[4, -1, 0],
              [1, 3, -1],
              [2, 0, 5]])

b = np.array([8, 6, 2])

# Parâmetros de controle
N = 100  # Número máximo de iterações
epsilon = 1e-5  # Critério de convergência

# Resolver o sistema de equações usando o método de Gauss-Seidel
solution = gauss_seidel(A, b, N, epsilon)

print("Solução:")
print(solution)
