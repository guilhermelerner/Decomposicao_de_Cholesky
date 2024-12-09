import numpy as np

def cholesky_decomposition(matrix):
    """
    Realiza a decomposição de Cholesky de uma matriz simétrica e definida positiva.

    Parâmetros:
        matrix (ndarray): Matriz simétrica e definida positiva.

    Retorna:
        ndarray: Matriz triangular inferior L da decomposição, onde matrix = L * L.T.
    """
    n = matrix.shape[0]
    lower_triangular = np.zeros_like(matrix)

    for row in range(n):
        for col in range(row + 1):
            if row == col:  # Elementos diagonais
                lower_triangular[row, col] = np.sqrt(matrix[row, row] - np.sum(lower_triangular[row, :col] ** 2))
            else:  # Elementos fora da diagonal
                lower_triangular[row, col] = (matrix[row, col] - np.sum(lower_triangular[row, :col] * lower_triangular[col, :col])) / lower_triangular[col, col]
    return lower_triangular

def solve_cholesky(matrix, vector):
    """
    Resolve o sistema linear Ax = b usando a decomposição de Cholesky.

    Parâmetros:
        matrix (ndarray): Matriz simétrica e definida positiva A.
        vector (ndarray): Vetor independente b.

    Retorna:
        ndarray: Solução x do sistema Ax = b.
    """
    lower_triangular = cholesky_decomposition(matrix)
    # Resolvendo Ly = b (substituição direta)
    intermediate_solution = np.linalg.solve(lower_triangular, vector)
    # Resolvendo L^T x = y (substituição retroativa)
    solution = np.linalg.solve(lower_triangular.T, intermediate_solution)
    return solution

def main():
    """
    Função principal para entrada de dados, validação e resolução do sistema Ax = b.
    """
    # Leitura da matriz A e do vetor b
    matrix_input = input("Digite a matriz A (ex: '1 2 3; 2 5 3; 3 3 6'): ")
    vector_input = input("Digite o vetor b (ex: '1; 2; 3'): ")

    # Processamento da entrada
    try:
        matrix = np.array([list(map(float, row.split())) for row in matrix_input.split(';')])
        vector = np.array(list(map(float, vector_input.split(';'))))
    except ValueError:
        print("Erro: certifique-se de que os valores inseridos estão no formato correto.")
        return

    # Verificando a compatibilidade da matriz e vetor
    if matrix.shape[0] != matrix.shape[1]:
        print("Erro: a matriz A deve ser quadrada.")
        return
    if matrix.shape[0] != vector.shape[0]:
        print("Erro: o vetor b deve ter o mesmo número de linhas que a matriz A.")
        return

    # Tentando resolver o sistema
    try:
        solution = solve_cholesky(matrix, vector)
        print("A solução do sistema é:", solution)
    except np.linalg.LinAlgError as error:
        print("Erro na decomposição de Cholesky:", error)

if __name__ == "__main__":
    main()
