import numpy as np
import pandas as pd
def curvature_A_matrix(n):
   
    #Construct the curvature A-matrix without Lawrence lifting.
    # Identity matrix of size (3n-1)
    idMat = np.eye(3 * n - 1, dtype=int)

    cols = []
    for i in range(n):
        for j in range(i, n):
            # Each column is a sum of three unit vectors
            col = idMat[i] + idMat[j] + idMat[n + i + j]
            cols.append(col)

    # make a matrix out of cols (and transpose)
    A = np.array(cols).T
    return A



def lawrence_lift(A):
    
    # Lawrence lifting of matrix A.
   
    A = np.array(A, dtype=int)
    c, d = A.shape

    idMat = np.eye(d, dtype=int)
    zeroMat = np.zeros((c, d), dtype=int)
    doubleIdMat = np.hstack((idMat, idMat))
    topMat = np.hstack((A, zeroMat))
    finalMat = np.vstack((topMat, doubleIdMat))
    return finalMat




    
    #design_mat = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
    #                   [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    #                   [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
    #                   [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
    #                   [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
    #                   [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])
    






# initial_states[0] = np.array([5, 0, 2, 1, 5, 1, 0, 0, 4, 1, 0, 0, 6, 0, 2, 0, 8, 0, 11, 0, 13, 0, 1, 0, 3, 0, 1, 0, 26, 0, 1, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 8, 2, 6, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 17, 10, 1, 1, 16, 7, 0, 0, 0, 2, 0, 0, 10, 6, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 3, 1, 1, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 3, 2, 0, 23, 4, 0, 0, 22, 2, 0, 0, 57, 3, 0, 0, 5, 1, 0, 0, 11, 0, 1, 0, 11, 0, 0, 0, 29, 2, 1, 1, 3, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 25, 0, 1, 37, 26, 0, 0, 15, 10, 0, 0, 43, 22, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 4, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




#=== Following function computes Joint degree matrix of input graph ===

def joint_degree_matrix(G):
    
    degrees = dict(G.degree())
    max_degree = max(degrees.values())

    # Initialize zero matrix
    jdm = np.zeros((max_degree+1, max_degree+1), dtype=int)

    # Count edges according to couple of degrees
    for u, v in G.edges():
        du, dv = degrees[u], degrees[v]
        jdm[du, dv] += 1
        if du != dv:
            jdm[dv, du] += 1  # jdm is symmetric

    MatJDM=jdm[1:,1:]

    # Return as a labeled DataFrame for readability
    return pd.DataFrame(MatJDM, index=range(max_degree), columns=range(max_degree))




def slack_matrix(matrix):
    max_degree=matrix.shape[1]
    sum_vec=np.sum(matrix, axis=1)+np.diag(matrix)
    degree_freq=np.array([sum_vec[i] // (i+1) for i in range(len(sum_vec))])
    M = np.outer(degree_freq, degree_freq)
    vals = degree_freq * (degree_freq - 1) // 2
    M[np.diag_indices(len(M))] = vals
    slackm= M-matrix
    return slackm



#Given a graph initialize a matrix 
def initialize_vector(G):
    matrix=joint_degree_matrix(G)
    slackm=slack_matrix(matrix)

    upper_entries_jdm = matrix.to_numpy()[np.triu_indices_from(matrix, k=0)]
    upper_entries_slackm = slackm.to_numpy()[np.triu_indices_from(slackm, k=0)]
    final_vec = np.concatenate((upper_entries_jdm, upper_entries_slackm))
    return final_vec





def vector_to_symmetric(v):
    # Determine the matrix size n from vector length
    # n(n+1)/2 = len(v)
    L = len(v)
    n = int((np.sqrt(8*L + 1) - 1) / 2)
    if n*(n+1)//2 != L:
        raise ValueError("Vector length is not valid for a symmetric matrix.")
    
    # Create empty symmetric matrix
    M = np.zeros((n, n), dtype=int)
    
    # Fill upper triangle (including diagonal) in lexicographic order
    idx = np.triu_indices(n)
    M[idx] = np.array(v, dtype=int)
    
    # Reflect upper triangle to lower triangle
    M = M + M.T - np.diag(np.diag(M))
    return M, n



def generate_jdm_15():
    # Initialize a 15x15 matrix filled with 1s (the off-diagonal connections)
    jdm = np.ones((15, 15), dtype=int)
    
    # These are the specific diagonal values J(i, i) that satisfy 
    # the row sum: 2*J(i, i) + 14 = i * n_i
    # for our chosen vertex distribution n_i.
    diagonal_values = [
        1,   # Degree 1
        1,   # Degree 2
        2,   # Degree 3
        1,   # Degree 4
        3,   # Degree 5
        2,   # Degree 6
        14,  # Degree 7
        1,   # Degree 8
        38,  # Degree 9
        43,  # Degree 10
        59,  # Degree 11
        65,  # Degree 12
        84,  # Degree 13
        91,  # Degree 14
        113  # Degree 15
    ]
    
    # Replace the diagonal with our calculated values
    np.fill_diagonal(jdm, diagonal_values)
    
    return jdm