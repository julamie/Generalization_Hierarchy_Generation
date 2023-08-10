import pandas as pd
import pm4py
import networkx as nx
import numpy as np

import Log_processing, Clustering

class Simple_Jaccard:
    def __init__(self, log):
        self.log = log
        
    def get_jaccard_distance_of_dfg(self):
        '''
        Computes the jaccard measure of two nodes using the DFG

        NetworkX provides an implementation of the jaccard measure in directly-follows-graphs.
        First the DFG has to be generated using pm4py. The output gets formatted for a NetworkX graph.
        Output is an iterator of the pairwise distances of the nodes in the graph
        '''

        # convert event log to an undirected NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        dg = nx.DiGraph()
        dg.add_weighted_edges_from(edge_list)

        # compute the jaccard measures of the nodes
        distances = nx.jaccard_coefficient(dg.to_undirected())

        return distances

    def convert_distances_to_distance_matrix(self, pairwise_distances):
        '''
        Converts the pairwise distances iterator from NetworkX to a distance matrix

        First finds the attributes and generates an empty distance matrix.
        The cells will be filled with 1 - the values of the corresponding jaccard measure.
        Remaining cells have the distance 1
        Returns the list of activities and the distance matrix
        '''
        
        # retrieve all attribute values
        activities = list(pm4py.get_event_attribute_values(self.log, "concept:name").keys())
        activities.sort()

        # create empty distance matrix using the activity names    
        distance_matrix = pd.DataFrame(columns=activities, index=activities)

        # fill matrix up
        for u, v, distance in pairwise_distances:
            distance_matrix[u][v] = 1 - round(distance, 6)
            distance_matrix[v][u] = 1 - round(distance, 6)

        distance_matrix = distance_matrix.fillna(1) # no similarity -> distance = 1
        distance_matrix = distance_matrix.to_numpy()
        np.fill_diagonal(distance_matrix, 0) # similarity to itself is always 0
        
        # return only the values of the matrix, not the names of the activities
        return activities, distance_matrix

    def perform_clustering(self, verbose=False, ax=None):
        '''
        Performs all necessary steps to perform hierarchical clustering using simple jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # generate a distance matrix using the directly follows graph of the events
        distances = self.get_jaccard_distance_of_dfg()
        activities, distance_matrix = self.convert_distances_to_distance_matrix(distances)
        
        # print the distances between events
        if verbose:
            display(distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        linkage = Clustering.create_linkage(distance_matrix)
        dendrogram = Clustering.create_dendrogram(activities, linkage, ax=ax)

        # generate the hierarchies dictionary
        clusterings = Clustering.create_clusterings_for_every_level(activities, distance_matrix, linkage)
        hierarchies = Clustering.create_hierarchy_for_activities(activities, clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(hierarchies)

        return dendrogram

class Weighted_Jaccard:
    def __init__(self, log):
        self.log = log

    def calculate_weighted_jaccard_similarity(self, G, node1, node2):
        '''
        Calculates the weighted jaccard distance between two nodes of a Graph
        Formula is the sum of the minimum weights of the union of vertices
        divited by the sum of the maximum weights of the union of vertices

        Source: https://stackoverflow.com/a/69218150
        '''
        
        # skip calculating the similarity if the nodes to compare are the same
        if node1 == node2:
            return 1

        neighbors1 = set(G.neighbors(node1))
        neighbors2 = set(G.neighbors(node2))
        minimums = 0
        maximums = 0

        for x in neighbors1.union(neighbors2):
            node1_weight = 0
            node2_weight = 0

            # get the weights of the nodes, if a node is not neighbouring x, then the weight is 0
            if x in G[node1]:
                node1_weight = G[node1][x]['weight']
            if x in G[node2]:
                node2_weight = G[node2][x]['weight']
            
            # sum up all the weights with minimal and maximal value
            minimums += min(node1_weight, node2_weight)
            maximums += max(node1_weight, node2_weight)
        # prevent division by 0 error
        if maximums == 0:
            return 0
        return minimums / maximums

    def get_weighted_jaccard_distance_matrix(self, connections_df):
        '''
        Computes the weighted jaccard distance for every node pair

        Returns a pivoted DataFrame where the row is the source node,
        the column the target node and the values are the distances between the nodes 
        '''
        
        G = nx.from_pandas_adjacency(connections_df, nx.DiGraph)
        jaccard_table = {} # the distances between the nodes

        # calculate the jaccard distances between all pairs of nodes and save them in jaccard_table
        for node1 in G.nodes:
            node1_distances = {}
            for node2 in G.nodes:
                similarity = self.calculate_weighted_jaccard_similarity(G, node1, node2)
                node1_distances[node2] = round(1 - similarity, 6)
            jaccard_table[node1] = node1_distances

        jaccard_table = pd.DataFrame(jaccard_table)
        return jaccard_table

    def perform_clustering(self, verbose=False, ax=None):
        '''
        Performs all necessary steps to perform hierarchical clustering using weighted jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''
        
        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # convert the log to a DataFrame with weights between the connections between events
        connections_df = Log_processing.get_pivot_df_from_dfg(self.log)
        connections_df = Log_processing.get_weighted_df(connections_df)

        # generate the weighted jaccard distance matrix
        distance_matrix = self.get_weighted_jaccard_distance_matrix(connections_df)
        activities = distance_matrix.columns
        
        # print the distances between events
        if verbose:
            display(distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        linkage = Clustering.create_linkage(distance_matrix.to_numpy())
        dendrogram = Clustering.create_dendrogram(activities, linkage, ax=ax)
        
        # generate the hierarchies dictionary
        clusterings = Clustering.create_clusterings_for_every_level(activities, distance_matrix, linkage)
        hierarchies = Clustering.create_hierarchy_for_activities(activities, clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(hierarchies)