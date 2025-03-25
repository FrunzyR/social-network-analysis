import requests
import gzip
import shutil
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"

file_path = "email-Eu-core.txt.gz"

response = requests.get(url)
if response.status_code == 200:
    with open(file_path, "wb") as file:
        file.write(response.content)
    print(f"File downloaded successfully as {file_path}")
else:
    print("Failed to download the file")

compressed_file = "email-Eu-core.txt.gz"
extracted_file = "email-Eu-core.txt"

with gzip.open("email-Eu-core.txt.gz", "rb") as f_in:
    with open("email-Eu-core.txt", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"File extracted successfully as {extracted_file}")

df = pd.read_csv("email-Eu-core.txt", header=None, names=["source", "target"])
print(df.head())

############ 1 FROM FIRST ASSIGMENT
# Number of Nodes (Unique Individuals)

num_nodes = pd.concat([df["source"], df["target"]]).nunique()
print(f"Number of unique nodes (email addresses): {num_nodes}")

# Number of Edges (Total Emails Sent)

num_edges = len(df)
print(f"Number of email interactions (edges): {num_edges}")

# Average Degree (Average Number of Connections per Node)

avg_degree = num_edges / num_nodes
print(f"Average degree (emails per user): {avg_degree:.2f}")


############ 2 FOR FIRST ASSIGMENT

G = nx.DiGraph()
G.add_edges_from(df.values)

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

in_degree_series = pd.Series(in_degree)
out_degree_series = pd.Series(out_degree)
total_degree_series = in_degree_series + out_degree_series

min_in_degree = in_degree_series.min()
max_in_degree = in_degree_series.max()
avg_in_degree = in_degree_series.mean()

min_out_degree = out_degree_series.min()
max_out_degree = out_degree_series.max()
avg_out_degree = out_degree_series.mean()

min_total_degree = total_degree_series.min()
max_total_degree = total_degree_series.max()
avg_total_degree = total_degree_series.mean()

print(f"Min In-Degree: {min_in_degree}")
print(f"Max In-Degree: {max_in_degree}")
print(f"Avg In-Degree: {avg_in_degree:.2f}")

print(f"Min Out-Degree: {min_out_degree}")
print(f"Max Out-Degree: {max_out_degree}")
print(f"Avg Out-Degree: {avg_out_degree:.2f}")

print(f"Min Total Degree: {min_total_degree}")
print(f"Max Total Degree: {max_total_degree}")
print(f"Avg Total Degree: {avg_total_degree:.2f}")

# Compute clustering coefficient
G = nx.DiGraph()
G.add_edges_from(df.values)

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
avg_clustering = nx.average_clustering(G)
print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

# Compute weakly connected components

wcc = list(nx.weakly_connected_components(G))
num_wcc = len(wcc)
largest_wcc_size = max(len(component) for component in wcc)

# Compute strongly connected components
scc = list(nx.strongly_connected_components(G))
num_scc = len(scc)
largest_scc_size = max(len(component) for component in scc)

# Print results
print(f"Number of Weakly Connected Components: {num_wcc}")
print(f"Size of Largest Weakly Connected Component: {largest_wcc_size}")

print(f"Number of Strongly Connected Components: {num_scc}")
print(f"Size of Largest Strongly Connected Component: {largest_scc_size}")

# Get the largest strongly connected component
largest_scc = max(scc, key=len)
G_largest_scc = G.subgraph(largest_scc)

# Compute diameter (only on largest SCC)
diameter = nx.diameter(G_largest_scc)
print(f"Network Diameter: {diameter}")

# Compute average shortest path length in the largest SCC
avg_shortest_path = nx.average_shortest_path_length(G_largest_scc)
print(f"Average Shortest Path Length: {avg_shortest_path:.2f}")


############ 3 FOR FIRST ASSIGMENT


############ 4 FOR FIRST ASSIGMENT

import seaborn as sns

# Degree Distribution
df = pd.read_csv("email-Eu-core.txt", sep=" ", header=None, names=["Sender", "Receiver"])

G = nx.DiGraph()
G.add_edges_from(df.values)

degree = dict(G.degree())

degree_series = pd.Series(degree)

plt.figure(figsize=(8, 5))
sns.histplot(degree_series, bins=50, kde=True, color="blue", log_scale=(True, True))
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.show()

# Clustering coefficient distribution
clustering_coeffs = nx.clustering(G)

clustering_series = pd.Series(clustering_coeffs)

plt.figure(figsize=(8, 5))
sns.histplot(clustering_series, bins=50, kde=True, color="red")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.title("📊 Clustering Coefficient Distribution")
plt.show()

# Betweenness Centrality Distribution
betweenness = nx.betweenness_centrality(G)

betweenness_series = pd.Series(betweenness)

plt.figure(figsize=(8, 5))
sns.histplot(betweenness_series, bins=50, kde=True, color="green", log_scale=(True, True))
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency")
plt.title("Betweenness Centrality Distribution")
plt.show()


# Compute weakly connected components
wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]

wcc_series = pd.Series(wcc_sizes)

plt.figure(figsize=(8, 5))
sns.histplot(wcc_series, bins=30, kde=True, color="purple", log_scale=(True, True))
plt.xlabel("Component Size (log scale)")
plt.ylabel("Frequency")
plt.title("🔗 Connected Components Size Distribution")
plt.show()



############ 5 FOR FIRST ASSIGMENT

# Compute degree centrality
degree_centrality = nx.degree_centrality(G)

# Get top 5 nodes with highest degree centrality
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Nodes by Degree Centrality:")
for node, value in top_degree:
    print(f"Node {node}: {value:.4f}")


# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Get top 5 nodes with highest betweenness centrality
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Nodes by Betweenness Centrality:")
for node, value in top_betweenness:
    print(f"Node {node}: {value:.4f}")


# Compute closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# Get top 5 nodes with highest closeness centrality
top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Nodes by Closeness Centrality:")
for node, value in top_closeness:
    print(f"Node {node}: {value:.4f}")


# Compute eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Get top 5 nodes with highest eigenvector centrality
top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Nodes by Eigenvector Centrality:")
for node, value in top_eigenvector:
    print(f"Node {node}: {value:.4f}")
