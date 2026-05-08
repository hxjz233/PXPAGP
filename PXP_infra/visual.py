import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import networkx as nx
from matplotlib.colors import LogNorm
import warnings
from utils import GetMeta

def PlotLevelDiff(param, E_diff_rescaled, r, plot_args=dict()):
    fig, ax = plt.subplots()

    # Plot the histogram of the rescaled differences with PDF
    bins = plot_args.get('bins_Ediff', 20)
    ax.hist(E_diff_rescaled, bins=bins, density=True, alpha=0.6, color='g')

    # Draw the reference line e^-x
    x = np.linspace(0, max(E_diff_rescaled), 100)
    poi = np.exp(-x)
    wd = np.pi * x / 2 * np.exp(-np.pi * x ** 2 / 4)
    ax.plot(x, poi, 'r--', label='Poisson')
    ax.plot(x, wd, 'b--', label='Wigner-Dyson')

    # Add labels and legend
    ax.set_xlabel('Rescaled Energy Differences')
    ax.set_ylabel('Probability Density')
    ax.set_title(GetMeta(param))
    ax.legend()

    # Add text for the number of data points
    num_data_points = len(E_diff_rescaled)
    ax.text(0.95, 0.95, f'N data points: {num_data_points}', 
            verticalalignment='top', horizontalalignment='right', 
            transform=ax.transAxes, color='black', fontsize=12)

    # Uncomment the following lines if you want to plot the histogram of r
    # ax.hist(r, bins=100, density=True, alpha=0.6, color='b')
    # ax.set_xlabel('r values')
    # ax.set_ylabel('Probability Density')

    return fig, ax


def PlotS_E(param, Emid, S):
    fig, ax = plt.subplots()

    # Adjust sizes of Emid and S
    if len(S) > len(Emid):
        warnings.warn("Size of S > size of Emid. Plotting only the first size(Emid) of S.")
        S = S[:len(Emid)]
    elif len(S) < len(Emid):
        warnings.warn("Size of S < size of Emid. Plotting only the first size(S) of Emid.")
        Emid = Emid[:len(S)]

    # Calculate the density
    xy = np.vstack([Emid, S])
    kde = gaussian_kde(xy)
    z = kde(xy)

    # Create scatter plot
    scatter = ax.scatter(Emid, S, c=z, s=50, cmap='turbo', norm=LogNorm())

    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Density (log scale)')

    # Add labels and title
    ax.set_xlabel('Energy (E)')
    ax.set_ylabel('Entropy (S)')
    ax.set_title(GetMeta(param))

    # Add text for the number of data points
    num_data_points = len(S)
    ax.text(0.95, 0.95, f'N data points: {num_data_points}', 
            verticalalignment='top', horizontalalignment='right', 
            transform=ax.transAxes, color='black', fontsize=12)

    return fig, ax

def PlotAdjacency(H):
    """
    Reads a complex adjacency matrix H and plots the corresponding graph.
    Uses color for the phase and width for the modulus of the complex numbers.
    
    Parameters:
    H (numpy.ndarray): Complex adjacency matrix representing the graph.
    
    Returns:
    fig, ax: Matplotlib figure and axis objects.
    """

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(np.abs(H))  # Use modulus for graph creation
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Draw the graph with adjusted spring_layout parameters
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust k and iterations for better layout

    # Extract edge attributes
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # print(edges)
    # print(weights)
    # phases = np.angle(H[np.triu_indices_from(H, k=1)])  # Get phases of upper triangular part
    phases = np.angle(H[np.tril_indices_from(H, k=-1)])  # Get phases of lower triangular part
    
    # Normalize weights and phases for plotting
    max_weight = max(weights)
    norm_weights = [weight / max_weight for weight in weights]
    norm_phases = [(phase + np.pi) / (2 * np.pi) for phase in phases]  # Normalize to [0, 1]
    
    # Draw edges with color and width based on phase and modulus
    for (u, weight, orig_phase, norm_phase) in zip(edges, norm_weights, phases, norm_phases):
        # print(f"Drawing edge ({u}) with weight {weight} and phase {orig_phase}")
        nx.draw_networkx_edges(G, pos, edgelist=[u], width=weight * 5, 
                               edge_color=plt.cm.hsv(norm_phase), ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Add color bar for phase
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Phase (radians)')
    
    # Add title
    ax.set_title('Graph from Complex Adjacency Matrix')
    
    return fig, ax

def PlotShist(param, S, plot_args=dict(), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        handles, labels = [], []
    else:
        fig = ax.figure
        handles, labels = ax.get_legend_handles_labels()

    # Plot the histogram of the entanglement entropies
    bins = plot_args.get('bins_Shist', 20)
    color = plot_args.get('color_Shist', 'g')
    ax.hist(S, bins=bins, density=True, alpha=0.6, color=color, label=plot_args.get('label_Shist', ''))

    # Add labels and title
    ax.set_xlabel('Entanglement Entropy (S)')
    ax.set_ylabel('Probability Density')
    ax.set_title(GetMeta(param))

    # Append to the legend
    label = plot_args.get('label_Shist', '')
    if label:
        labels.append(label)
        handles.append(ax.patches[-1])
    ax.legend(handles, labels)

    # Add text for the number of data points, adjusting position to avoid overlap
    num_data_points = len(S)
    existing_texts = len(ax.texts)
    ax.text(0.95, 0.95 - 0.05 * existing_texts, f'N data points: {num_data_points}', 
            verticalalignment='top', horizontalalignment='right', 
            transform=ax.transAxes, color='black', fontsize=12)

    return fig, ax