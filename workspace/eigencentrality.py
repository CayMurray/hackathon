import numpy as np
import matplotlib.pyplot as plt
from funcs import load_labels
import networkx as nx
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


data = np.load('./connectomes_cobre_scale_444.npy')


labels = load_labels("./subjects.txt")
labels = np.array(labels["labels"])

data_cont = data[:, :, labels == "cont"]
data_sz = data[:, :, labels == "sz"]

data_avg_cont = np.mean(np.abs(data_cont), axis=2)
data_avg_sz = np.mean(np.abs(data_sz), axis=2)


# Create a network from a correlation matrix with a threshold
def create_network(corr_matrix, threshold=0.8):
    np.fill_diagonal(corr_matrix, 0)  # Remove self-loops
    corr_matrix[np.abs(corr_matrix) < threshold] = 0  # Apply threshold

    G = nx.from_numpy_array(corr_matrix)
    return G

G_cont = create_network(data_avg_cont, threshold=0.8)
G_scz = create_network(data_avg_sz, threshold=0.8)

deg_cent_cont = nx.degree_centrality(G_cont)
deg_cent_scz = nx.degree_centrality(G_scz)


eigenvector_cont = nx.eigenvector_centrality_numpy(G_cont, max_iter=1000, weight='weight')
eigenvector_scz = nx.eigenvector_centrality_numpy(G_scz, max_iter=1000, weight='weight')


# Define node positions based on the control group for consistent layout across both networks
pos = nx.spring_layout(G_cont)

# Prepare the plot
plt.figure(figsize=(10, 10))
plt.title("Eigenvector Centrality: Control (Blue) vs Schizophrenia (Orange)")

# Draw Control Network
nx.draw_networkx_nodes(G_cont, pos, node_color='orange', node_size=[800 * np.abs(v) for v in eigenvector_cont.values()], label='Control')
nx.draw_networkx_edges(G_cont, pos, edge_color='orange', alpha=1)

# Draw Schizophrenia Network
nx.draw_networkx_nodes(G_scz, pos, node_color='blue', node_size=[800 * np.abs(v) for v in eigenvector_scz.values()], label='Schizophrenia')
nx.draw_networkx_edges(G_scz, pos, edge_color='blue', alpha=0.5)

# Show legend with small markers
plt.legend(markerscale=0.5)

# Hide axes
plt.axis('off')
plt.savefig('network_comparison.png')


def compute_metrics(graph):
    metrics = {
        'degree': dict(nx.degree_centrality(graph)),
        'eigenvector': dict(nx.eigenvector_centrality(graph, max_iter=1000, weight='weight')),
        'betweenness': dict(nx.betweenness_centrality(graph, weight='weight'))
    }
    return metrics

metrics_cont = compute_metrics(G_cont)
metrics_scz = compute_metrics(G_scz)

def extract_values(metrics):
    return {key: list(val.values()) for key, val in metrics.items()}

values_cont = extract_values(metrics_cont)
values_scz = extract_values(metrics_scz)

# Two-sided Mann-Whitney U test for each metric
p_values_left = []
test_results_left = {}

for metric in values_cont:
    stat, p = mannwhitneyu(values_cont[metric], values_scz[metric], alternative='two-sided')
    test_results_left[metric] = (stat, p)
    p_values_left.append(p)

# FDR correction
rejected_left, p_values_corrected_left, _, _ = multipletests(p_values_left, alpha=0.05, method='fdr_by')

for i, metric in enumerate(values_cont):
    print(f"Two-sided Mann-Whitney U Test for {metric.capitalize()}:\n"
          f"U Statistic: {test_results_left[metric][0]}, "
          f"Median Control: {np.median(values_cont[metric])}, "
          f"Median Schizophrenia: {np.median(values_scz[metric])}, "
          f"P-Value: {test_results_left[metric][1]}, "
          f"Corrected P-Value: {p_values_corrected_left[i]}, "
          f"Reject Null: {rejected_left[i]}")
    
p_values_left = []
test_results_left = {}
print("\n")
# Left-sided Mann-Whitney U test
for metric in values_cont:
    stat, p = mannwhitneyu(values_cont[metric], values_scz[metric], alternative='less')
    test_results_left[metric] = (stat, p)
    p_values_left.append(p)

# Apply FDR correction
rejected_left, p_values_corrected_left, _, _ = multipletests(p_values_left, alpha=0.05, method='fdr_by')

for i, metric in enumerate(values_cont):
    print(f"Left-sided Mann-Whitney U Test for {metric.capitalize()}:\n"
          f"U Statistic: {test_results_left[metric][0]}, "
          f"Median Control: {np.median(values_cont[metric])}, "
          f"Median Schizophrenia: {np.median(values_scz[metric])}, "
          f"P-Value: {test_results_left[metric][1]}, "
          f"Corrected P-Value: {p_values_corrected_left[i]}, "
          f"Reject Null: {rejected_left[i]}")
    
    
