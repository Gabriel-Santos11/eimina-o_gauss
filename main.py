import numpy as np

def gauss_elimination(A, b):

    Ab = np.column_stack((A, b))

    #Gauss
    n = len(Ab)
    for i in range(n):

        pivot_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, pivot_row]] = Ab[[pivot_row, i]]

        pivot = Ab[i, i]
        Ab[i] = Ab[i] / pivot

        for j in range(i + 1, n):
            Ab[j] -= Ab[i] * Ab[j, i]


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - Ab[i, :-1] @ x

    return x

# Sistema
A = np.array([[4, 3, -1],
              [9, -2, 7],
              [3, 2, 1]])

b = np.array([3, -2, 7])

# Execução do Gauss
solution = gauss_elimination(A, b)

print("Solução:")
print(solution)