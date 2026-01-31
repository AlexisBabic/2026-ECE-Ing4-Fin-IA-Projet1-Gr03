"""
Graph Coloring Problem Implementation

This module implements the graph coloring problem using Constraint Programming
with OR-Tools. It assigns colors to graph nodes such that no two adjacent nodes
share the same color, minimizing the number of colors used.

Author: Groupe 01 IA Finance
"""

import networkx as nx
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt


def create_sample_graph():
    """
    Create a sample graph representing a map with regions to color.
    This is a simple graph with 5 nodes forming a pentagon plus a center.

    Returns:
        networkx.Graph: The graph to color
    """
    G = nx.Graph()
    # Add nodes (regions)
    regions = ['A', 'B', 'C', 'D', 'E', 'Center']
    G.add_nodes_from(regions)

    # Add edges (adjacencies)
    edges = [
        ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A'),  # pentagon
        ('Center', 'A'), ('Center', 'B'), ('Center', 'C'), ('Center', 'D'), ('Center', 'E')  # center connected to all
    ]
    G.add_edges_from(edges)

    return G


def create_random_graph(n=8, p=0.3, seed=42):
    """
    Create a random graph using Erdős–Rényi model.

    Args:
        n (int): Number of nodes
        p (float): Probability of edge creation
        seed (int): Random seed

    Returns:
        networkx.Graph: Random graph
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Label nodes as strings
    mapping = {i: f'Node_{i}' for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    return G


def create_dense_graph(n=6):
    """
    Create a dense graph (complete graph minus some edges).

    Args:
        n (int): Number of nodes

    Returns:
        networkx.Graph: Dense graph
    """
    G = nx.complete_graph(n)
    # Label nodes first
    mapping = {i: f'Dense_{i}' for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    # Then remove some edges
    edges_to_remove = [('Dense_0', 'Dense_3'), ('Dense_1', 'Dense_4'), ('Dense_2', 'Dense_5')]
    G.remove_edges_from(edges_to_remove)
    return G


def solve_graph_coloring(G, max_colors=4):
    """
    Solve the graph coloring problem using CP-SAT with minimization objective.

    Args:
        G (networkx.Graph): The graph to color
        max_colors (int): Maximum number of colors to use

    Returns:
        dict: Mapping of node to color if solution found, None otherwise
        int: Number of colors used
    """
    model = cp_model.CpModel()
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Heuristic: sort nodes by degree descending (conceptual, as CP-SAT handles search internally)
    nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)

    # Create color variables for each node
    colors = {}
    for node in nodes_sorted:
        colors[node] = model.NewIntVar(0, max_colors - 1, f'color_{node}')

    # Symmetry breaking: fix first node's color to 0
    if nodes_sorted:
        model.Add(colors[nodes_sorted[0]] == 0)

    # Add constraints: adjacent nodes must have different colors
    for u, v in G.edges():
        model.Add(colors[u] != colors[v])

    # Objective: minimize the maximum color used
    max_color_used = model.NewIntVar(0, max_colors - 1, 'max_color_used')
    for node in nodes:
        model.Add(colors[node] <= max_color_used)
    model.Minimize(max_color_used)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = {node: solver.Value(colors[node]) for node in nodes}
        num_colors_used = solver.Value(max_color_used) + 1  # +1 since colors start at 0
        return solution, num_colors_used
    else:
        return None, None


def greedy_coloring(G, strategy='largest_first', max_colors=10):
    """
    Implement greedy graph coloring.

    Args:
        G (networkx.Graph): The graph to color
        strategy (str): Ordering strategy ('largest_first' or 'random')
        max_colors (int): Maximum colors available

    Returns:
        dict: Node to color mapping
        int: Number of colors used
    """
    if strategy == 'largest_first':
        # Sort nodes by degree descending
        nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    else:
        nodes = list(G.nodes())

    coloring = {}
    available_colors = set(range(max_colors))

    for node in nodes:
        # Find colors used by neighbors
        neighbor_colors = {coloring.get(neigh) for neigh in G.neighbors(node) if neigh in coloring}
        # Assign smallest available color
        for color in available_colors:
            if color not in neighbor_colors:
                coloring[node] = color
                break

    num_colors = len(set(coloring.values()))
    return coloring, num_colors


def visualize_graph(G, coloring=None, ax=None, title=None):
    """
    Visualize the graph with colors.

    Args:
        G (networkx.Graph): The graph
        coloring (dict): Node to color mapping
        ax (matplotlib.axes.Axes): Axes to plot on (for subplots)
        title (str): Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Position nodes
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout

    # Fixed color palette
    color_palette = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']

    # Color map
    if coloring:
        node_colors = [color_palette[coloring.get(node, 0) % len(color_palette)] for node in G.nodes()]
    else:
        node_colors = 'lightgray'

    # Draw graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='black',
            node_size=500, font_size=16, font_weight='bold', ax=ax)

    if title:
        ax.set_title(title)

    # Add legend if coloring
    if coloring:
        unique_colors = sorted(set(coloring.values()))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_palette[c % len(color_palette)],
                                      markersize=10, label=f'Color {c}') for c in unique_colors]
        ax.legend(handles=legend_elements, loc='upper right')


def create_performance_chart():
    """
    Create a performance comparison chart for different graph sizes.
    """
    import time
    import matplotlib.pyplot as plt

    # Test different graph sizes (from 10 to 100 nodes for reasonable completion time)
    sizes = [10, 20, 50, 100]
    cp_times = []
    greedy_times = []
    cp_colors_used = []
    greedy_colors_used = []

    max_colors = 50  # Increased for larger graphs
    runs_per_size = 3  # Multiple runs for averaging

    print("Performance Testing Across Graph Sizes")
    print("=" * 40)

    for n in sizes:
        print(f"Testing graph with {n} nodes...")

        # Adaptive edge probability and runs for large graphs
        if n <= 50:
            p = 0.1   # Lower density for smaller graphs
            runs = 3
        elif n <= 100:
            p = 0.05  # Lower density for medium graphs
            runs = 2
        else:
            p = 0.02  # Very sparse for large graphs
            runs = 1  # Single run for large graphs

        # Create a random graph for this size
        G = create_random_graph(n=n, p=p, seed=42)

        # Test CP
        cp_time_total = 0
        cp_colors_list = []
        for _ in range(runs):
            start = time.time()
            cp_coloring, cp_colors = solve_graph_coloring(G, max_colors=max_colors)
            cp_time_total += time.time() - start
            cp_colors_list.append(cp_colors)

        avg_cp_time = cp_time_total / runs
        avg_cp_colors = sum(cp_colors_list) / runs

        cp_times.append(avg_cp_time)
        cp_colors_used.append(avg_cp_colors)

        # Test Greedy
        greedy_time_total = 0
        greedy_colors_list = []
        for _ in range(runs):
            start = time.time()
            greedy_coloring_result, greedy_colors = greedy_coloring(G, strategy='largest_first', max_colors=max_colors)
            greedy_time_total += time.time() - start
            greedy_colors_list.append(greedy_colors)

        avg_greedy_time = greedy_time_total / runs
        avg_greedy_colors = sum(greedy_colors_list) / runs

        greedy_times.append(avg_greedy_time)
        greedy_colors_used.append(avg_greedy_colors)

        print(f"  CP: {avg_cp_time:.4f}s, {avg_cp_colors:.1f} colors")
        print(f"  Greedy: {avg_greedy_time:.6f}s, {avg_greedy_colors:.1f} colors")

    # Create the performance chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time comparison
    ax1.plot(sizes, cp_times, 'o-', label='CP-SAT', color='blue', linewidth=2, markersize=8)
    ax1.plot(sizes, greedy_times, 's-', label='Greedy', color='red', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Algorithm Performance: Time vs Graph Size (10-100 nodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    ax1.set_xscale('log')  # Log scale for x-axis too

    # Colors used comparison
    ax2.plot(sizes, cp_colors_used, 'o-', label='CP-SAT', color='blue', linewidth=2, markersize=8)
    ax2.plot(sizes, greedy_colors_used, 's-', label='Greedy', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Colors Used')
    ax2.set_title('Solution Quality: Colors Used vs Graph Size (10-100 nodes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')  # Log scale for x-axis

    plt.tight_layout()
    plt.savefig('algorithm_performance_comparison_10_to_100.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance chart saved as 'algorithm_performance_comparison_10_to_100.png'")

    # Also save data to CSV
    import csv
    with open('performance_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['nodes', 'cp_time', 'greedy_time', 'cp_colors', 'greedy_colors'])
        for i, n in enumerate(sizes):
            writer.writerow([n, cp_times[i], greedy_times[i], cp_colors_used[i], greedy_colors_used[i]])
    
    print(f"Performance data also saved to 'performance_data.csv'")

    return sizes, cp_times, greedy_times, cp_colors_used, greedy_colors_used


def main():
    """
    Main function to run the graph coloring example on multiple graphs.
    """
    print("Graph Coloring Problem - Multi-Graph Testing")
    print("=" * 50)
    print("Note: Graph coloring is NP-difficile; the 4-color theorem states that planar graphs need at most 4 colors.")

    # First, create performance comparison chart
    print("\nGenerating performance comparison chart...")
    create_performance_chart()

    # Then run individual graph tests
    graphs = [
        ("Sample Graph", create_sample_graph()),
        ("Random Graph (8 nodes)", create_random_graph(n=8, p=0.4, seed=42)),
        ("Dense Graph (6 nodes)", create_dense_graph(n=6)),
        ("Large Random Graph (50 nodes)", create_random_graph(n=50, p=0.1, seed=42))
    ]

    max_colors = 20  # Increased for larger graphs

    for name, G in graphs:
        print(f"\n--- Testing on {name} ---")
        print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

        # Adaptive runs: fewer for larger graphs
        num_runs = 1 if len(G.nodes()) > 20 else 5

        # CP Coloring - average over runs
        cp_times = []
        cp_results = []
        for _ in range(num_runs):
            import time
            start = time.time()
            cp_coloring, cp_colors = solve_graph_coloring(G, max_colors=max_colors)
            cp_times.append(time.time() - start)
            cp_results.append((cp_coloring, cp_colors))

        avg_cp_time = sum(cp_times) / num_runs
        # Use the result from the last run for visualization
        cp_coloring, cp_colors = cp_results[-1]

        if cp_coloring:
            print(f"CP: {cp_colors} colors, Avg time: {avg_cp_time:.4f}s")
        else:
            print("CP: No solution found")

        # Greedy Coloring - average over runs
        greedy_times = []
        greedy_results = []
        for _ in range(num_runs):
            start = time.time()
            greedy_coloring_result, greedy_colors = greedy_coloring(G, strategy='largest_first', max_colors=max_colors)
            greedy_times.append(time.time() - start)
            greedy_results.append((greedy_coloring_result, greedy_colors))

        avg_greedy_time = sum(greedy_times) / num_runs
        greedy_coloring_result, greedy_colors = greedy_results[-1]

        print(f"Greedy: {greedy_colors} colors, Avg time: {avg_greedy_time:.4f}s")

        # Visualize comparison: CP vs Greedy (skip for large graphs)
        if cp_coloring and greedy_coloring_result:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            visualize_graph(G, ax=ax1, title=f"{name} - Uncolored")
            visualize_graph(G, cp_coloring, ax=ax2, title=f"CP ({cp_colors} colors, {avg_cp_time:.4f}s)")
            visualize_graph(G, ax=ax3, title=f"{name} - Uncolored (again)")
            visualize_graph(G, greedy_coloring_result, ax=ax4, title=f"Greedy ({greedy_colors} colors, {avg_greedy_time:.4f}s)")
            plt.tight_layout()
            filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('nodes', 'nodes')}_comparison.png"
            plt.savefig(filename)
            print(f"Comparison plot saved to {filename}")
            plt.close()
        else:
            print("Visualization skipped due to missing solutions.")


if __name__ == "__main__":
    main()