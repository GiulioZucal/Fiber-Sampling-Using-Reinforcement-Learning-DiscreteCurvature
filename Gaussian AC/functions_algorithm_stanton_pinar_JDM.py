from collections import defaultdict, deque
import heapq
from matrix_utils import *


def havel_hakimi(degrees, nodes):
    """
    Construct a simple graph using the Havel-Hakimi algorithm.
    degrees: list of degrees
    nodes: list of node indices corresponding to degrees
    Returns list of edges
    """
    edges = []
    heap = [(-deg, node) for deg, node in zip(degrees, nodes)]
    heapq.heapify(heap)

    while heap:
        deg, u = heapq.heappop(heap)
        deg = -deg
        if deg == 0:
            break

        if deg > len(heap):
            raise ValueError("Invalid degree sequence")

        temp = []
        for _ in range(deg):
            d2, v = heapq.heappop(heap)
            d2 = -d2
            edges.append((u, v))
            if d2 - 1 > 0:
                temp.append((-(d2 - 1), v))

        for item in temp:
            heapq.heappush(heap, item)

    return edges


def build_bipartite_graph(left_nodes, right_nodes, left_deg, right_deg):
    """
    Construct a simple bipartite graph using the Bipartite Havel-Hakimi / Gale-Ryser algorithm.
    """
    edges = []
    
    # Create lists of lists: [remaining_degree, node_id]
    left = [[d, n] for d, n in zip(left_deg, left_nodes) if d > 0]
    right = [[d, n] for d, n in zip(right_deg, right_nodes) if d > 0]

    # Sort left side descending to process highest degrees first
    left.sort(key=lambda x: x[0], reverse=True)

    for i in range(len(left)):
        l_deg, u = left[i]

        # Sort right side descending by remaining degree to avoid parallel edges
        # and maximize the chances of fulfilling the degree sequence.
        right.sort(key=lambda x: x[0], reverse=True)

        if l_deg > len(right):
            raise ValueError(f"Cannot form simple bipartite graph: left degree {l_deg} exceeds available right nodes {len(right)}.")

        # Connect to the top 'l_deg' nodes on the right
        for j in range(l_deg):
            if right[j][0] <= 0:
                raise ValueError("Not enough available edges on the right side.")
            
            edges.append((u, right[j][1]))
            right[j][0] -= 1  # Decrement the remaining degree

        # Clean up fully connected right nodes to speed up the next sort
        right = [r for r in right if r[0] > 0]

    return edges

from collections import defaultdict, deque
import heapq
import numpy as np

def construct_graph(J, D):
    """
    Implements Algorithm 1 using NumPy inputs.
    J: 2D NumPy array (Joint Degree Matrix)
    D: 1D NumPy array (Degree Distribution Vector)
    """
    # Step 1: Create vertices grouped by degree
    # Map: degree k = index + 1
    nodes_by_degree = defaultdict(list)
    node_id = 0
    
    # Iterate backwards to match sorted(reverse=True)
    for i in range(len(D) - 1, -1, -1):
        k = i + 1  # Actual degree
        count = int(D[i])
        for _ in range(count):
            nodes_by_degree[k].append(node_id)
            node_id += 1

    # Initialize residual degrees
    residual = {}
    for k, nodes in nodes_by_degree.items():
        for v in nodes:
            residual[v] = k

    G = defaultdict(set)

    # Main loop through the Joint Degree Matrix J
    # i corresponds to degree k, j corresponds to degree l
    for i in range(len(D) - 1, -1, -1):
        for j in range(len(D) - 1, -1, -1):
            k, l = i + 1, j + 1
            
            # Use NumPy indexing to get the number of edges between degrees k and l
            val_j = J[i, j]
            
            if l > k or val_j == 0:
                continue

            Dk = len(nodes_by_degree[k])
            Dl = len(nodes_by_degree[l])
            
            if Dk == 0 or Dl == 0:
                continue

            if k != l:
                # Step 3: Bipartite construction
                a = val_j % Dk
                b = val_j % Dl

                base_k = val_j // Dk
                base_l = val_j // Dl

                # Sort nodes by current residual degree (descending)
                left_nodes = sorted(nodes_by_degree[k], key=lambda v: -residual[v])
                right_nodes = sorted(nodes_by_degree[l], key=lambda v: -residual[v])

                left_deg = [int(base_k + 1) if idx < a else int(base_k) for idx in range(Dk)]
                right_deg = [int(base_l + 1) if idx < b else int(base_l) for idx in range(Dl)]

                B_edges = build_bipartite_graph(left_nodes, right_nodes, left_deg, right_deg)

            else:
                # Case k == l: Simple graph construction
                # We multiply by 2 because J[k,k] usually counts edges, 
                # but the degree sum is 2 * edges.
                c = int(2 * val_j) % Dk
                base = int(2 * val_j) // Dk

                nodes = sorted(nodes_by_degree[k], key=lambda v: -residual[v])
                deg_seq = [int(base + 1) if idx < c else int(base) for idx in range(Dk)]

                B_edges = havel_hakimi(deg_seq, nodes)

            # Insert into global graph and update residual degrees
            for u, v in B_edges:
                if u == v:
                    continue
                if v not in G[u]:
                    G[u].add(v)
                    G[v].add(u)
                    residual[u] -= 1
                    residual[v] -= 1

    return G