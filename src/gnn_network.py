"""
Graph Neural Network for customer geographic network analysis.
Models regional influence on churn decisions using network analysis and node embeddings.
"""

import logging
import os
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeographicNetworkBuilder:
    """
    Builds customer geographic network and computes network features.
    
    Network representation:
    - Nodes: Customers
    - Edges: Geographic proximity + service similarity + demographic similarity
    
    Captures network effects on churn through:
    - Node centrality (influence in network)
    - Local churn rates (peer effects)
    - Community structure (regional clusters)
    """
    
    def __init__(self, distance_threshold: float = 50, similarity_threshold: float = 0.5):
        """
        Initialize network builder.
        
        Args:
            distance_threshold: Geographic distance threshold for connecting nodes
            similarity_threshold: Minimum similarity score for edge creation
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.graph = None
        self.embeddings = None
        self.logger = logging.getLogger(__name__)
    
    def build_network(self, df: pd.DataFrame, region_column: Optional[str] = None) -> nx.Graph:
        """
        Build network graph based on customer similarities.
        
        Connection criteria:
        1. Geographic proximity (same ZIP code region)
        2. Service similarity (shared services)
        3. Demographic similarity (age, family status)
        
        Args:
            df: Customer dataframe
            region_column: Column containing region/ZIP code information
        
        Returns:
            NetworkX graph object
        """
        
        self.graph = nx.Graph()
        
        # Add all customers as nodes
        for idx in df.index:
            self.graph.add_node(idx)
        
        self.logger.info(f"Added {self.graph.number_of_nodes()} nodes to graph")
        
        # Group customers by region if available
        if region_column is not None and region_column in df.columns:
            # Use first 3 characters of ZIP code for regional grouping
            regions = df.groupby(df[region_column].astype(str).str[:3]).groups
            self.logger.info(f"Found {len(regions)} regions")
            
            edge_count = 0
            for region, customers in regions.items():
                # Connect customers within same region
                customers_list = list(customers)
                for i, cust1 in enumerate(customers_list[:-1]):
                    for cust2 in customers_list[i+1:]:
                        weight = self._calculate_edge_weight(df, cust1, cust2)
                        if weight > 0:
                            self.graph.add_edge(cust1, cust2, weight=weight)
                            edge_count += 1
        else:
            # No region info: fall back to an efficient grouping strategy to avoid O(n^2) complexity.
            self.logger.info("No region column provided. Using proxy grouping for efficient network construction.")
            
            # Group by key categorical features to create smaller comparison pools.
            grouping_cols = ['Contract', 'InternetService', 'PaymentMethod']
            valid_grouping_cols = [col for col in grouping_cols if col in df.columns]
            
            if not valid_grouping_cols:
                self.logger.error("Cannot perform proxy grouping. At least one of 'Contract', 'InternetService', 'PaymentMethod' must be in the dataframe.")
                raise ValueError("Cannot build network without region or valid grouping columns.")

            groups = df.groupby(valid_grouping_cols).groups
            self.logger.info(f"Created {len(groups)} proxy groups for pairwise comparison.")

            edge_count = 0
            try:
                from tqdm import tqdm
                iterator = tqdm(groups.items(), desc="Building edges within groups")
            except ImportError:
                iterator = groups.items()
            
            for group, customers in iterator:
                customers_list = list(customers)
                # Iterate through pairs within the smaller group
                for i in range(len(customers_list)):
                    for j in range(i + 1, len(customers_list)):
                        cust1 = customers_list[i]
                        cust2 = customers_list[j]
                        weight = self._calculate_edge_weight(df, cust1, cust2)
                        if weight > 0:
                            self.graph.add_edge(cust1, cust2, weight=weight)
                            edge_count += 1
        
        self.logger.info(f"Added {edge_count} edges to graph")
        self.logger.info(f"Graph density: {nx.density(self.graph):.4f}")
        
        return self.graph
    
    def _calculate_edge_weight(self, df: pd.DataFrame, cust1_idx: int, cust2_idx: int) -> float:
        """
        Calculate edge weight between two customers.
        
        Weight combines:
        - Service similarity (0.5 weight): shared services create bundling opportunity
        - Demographic similarity (0.3 weight): similar profiles influence each other
        - Contract similarity (0.2 weight): similar commitments
        
        Args:
            df: Customer dataframe
            cust1_idx: First customer index
            cust2_idx: Second customer index
        
        Returns:
            Edge weight (0.0 to 1.0)
        """
        
        try:
            # Service similarity: cosine similarity on service adoption
            service_cols = [col for col in df.columns if 'Service' in col or 'service' in col]
            if service_cols:
                services_cust1 = df.loc[cust1_idx, service_cols].values.astype(float)
                services_cust2 = df.loc[cust2_idx, service_cols].values.astype(float)
                
                # Cosine similarity
                dot_product = np.dot(services_cust1, services_cust2)
                norm1 = np.linalg.norm(services_cust1)
                norm2 = np.linalg.norm(services_cust2)
                
                if norm1 > 0 and norm2 > 0:
                    service_sim = dot_product / (norm1 * norm2)
                else:
                    service_sim = 0.0
            else:
                service_sim = 0.0
            
            # Demographic similarity: binary features similarity
            demo_cols = ['SeniorCitizen', 'Partner', 'Dependents']
            demo_cols_present = [col for col in demo_cols if col in df.columns]
            
            if demo_cols_present:
                demo_cust1 = df.loc[cust1_idx, demo_cols_present].values.astype(float)
                demo_cust2 = df.loc[cust2_idx, demo_cols_present].values.astype(float)
                
                # Jaccard similarity for binary features
                intersection = np.sum(demo_cust1 == demo_cust2)
                union = len(demo_cols_present)
                demo_sim = intersection / union if union > 0 else 0.0
            else:
                demo_sim = 0.0
            
            # Contract similarity
            if 'Contract' in df.columns:
                contract_sim = 1.0 if df.loc[cust1_idx, 'Contract'] == df.loc[cust2_idx, 'Contract'] else 0.5
            else:
                contract_sim = 0.5
            
            # Weighted combination
            weight = (0.5 * service_sim + 0.3 * demo_sim + 0.2 * contract_sim)
            
            return weight if weight > self.similarity_threshold else 0.0
        
        except Exception as e:
            self.logger.warning(f"Error calculating weight between {cust1_idx} and {cust2_idx}: {e}")
            return 0.0
    
    def compute_node2vec_embeddings(self, dimensions: int = 16, walks_per_node: int = 10) -> pd.DataFrame:
        """
        Compute node embeddings using Node2Vec algorithm.
        
        Captures network topology in lower-dimensional space.
        Nodes with similar network positions get similar embeddings.
        
        Args:
            dimensions: Embedding dimension
            walks_per_node: Number of random walks per node
        
        Returns:
            DataFrame with embeddings
        """
        
        if self.graph is None:
            raise ValueError("Network not built. Call build_network() first.")
        
        try:
            from node2vec import Node2Vec
        except ImportError:
            self.logger.error("node2vec not installed. Install with: pip install node2vec")
            raise
        
        self.logger.info(f"Computing Node2Vec embeddings (dim={dimensions}, walks={walks_per_node})...")

        # Windows can fail when Node2Vec tries to pickle work to multiple processes.
        # Use a single worker for compatibility in this notebook environment.
        workers = 1 if os.name == "nt" else max(1, min(4, os.cpu_count() or 1))
        self.logger.info(f"Using Node2Vec workers={workers}")
        
        try:
            node2vec = Node2Vec(
                self.graph,
                dimensions=dimensions,
                walk_length=30,
                num_walks=walks_per_node,
                workers=workers,
                seed=42
            )
            
            model = node2vec.fit(window=10, min_count=1, batch_words=4, epochs=1)
            
            # Extract embeddings
            embeddings_dict = {}
            for node in self.graph.nodes():
                try:
                    embeddings_dict[node] = model.wv[node]
                except KeyError:
                    # Some nodes might not have embeddings
                    embeddings_dict[node] = np.random.randn(dimensions)
            
            self.embeddings = pd.DataFrame(embeddings_dict).T
            self.embeddings.columns = [f'gnn_embedding_{i}' for i in range(dimensions)]
            
            self.logger.info(f"Embeddings computed: shape {self.embeddings.shape}")
            return self.embeddings
        
        except Exception as e:
            self.logger.error(f"Error computing embeddings: {e}")
            raise
    
    def compute_network_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute network-based features for each customer.
        
        Features:
        - Degree centrality: how connected a customer is
        - Betweenness centrality: how much a customer bridges communities
        - Clustering coefficient: how much neighbors are connected
        - Local churn rate: average churn of neighbors
        - Community detection: which community a customer belongs to
        
        Args:
            df: Customer dataframe (must have 'Churn' column)
        
        Returns:
            DataFrame with network metrics
        """
        
        if self.graph is None:
            raise ValueError("Network not built. Call build_network() first.")
        
        self.logger.info("Computing network metrics...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Degree Centrality: proportion of nodes connected to
        degree_centrality = nx.degree_centrality(self.graph)
        features['network_degree_centrality'] = pd.Series(degree_centrality)
        
        # 2. Betweenness Centrality: importance in network paths
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            features['network_betweenness_centrality'] = pd.Series(betweenness)
        except Exception as e:
            self.logger.warning(f"Could not compute betweenness centrality: {e}")
            features['network_betweenness_centrality'] = 0.0
        
        # 3. Clustering Coefficient: local network density
        clustering = nx.clustering(self.graph)
        features['network_clustering_coeff'] = pd.Series(clustering)
        
        # 4. Local churn rate: average churn of neighbors
        local_churn = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                local_churn[node] = df.loc[neighbors, 'Churn'].mean()
            else:
                local_churn[node] = df['Churn'].mean()
        
        features['network_local_churn_rate'] = pd.Series(local_churn)
        
        # 5. Network size (neighbors)
        neighbors_count = {}
        for node in self.graph.nodes():
            neighbors_count[node] = len(list(self.graph.neighbors(node)))
        
        features['network_neighbors_count'] = pd.Series(neighbors_count)
        
        # 6. Community detection using Louvain method
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
            features['network_community'] = pd.Series(communities)
        except ImportError:
            self.logger.warning("python-louvain not installed. Skipping community detection.")
            features['network_community'] = 0
        except Exception as e:
            self.logger.warning(f"Could not detect communities: {e}")
            features['network_community'] = 0
        
        self.logger.info(f"Network metrics computed: {features.columns.tolist()}")
        return features
    
    def get_network_statistics(self) -> Dict:
        """
        Get summary statistics about the network.
        
        Returns:
            Dictionary with network statistics
        """
        
        if self.graph is None:
            return {}
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'average_clustering': nx.average_clustering(self.graph),
        }
        
        return stats
