import pandas as pd
import pm4py
import networkx as nx
import numpy as np

import Log_processing, Clustering

class Simple_Jaccard:
    def __init__(self, log):
        self.log = log
        self.distances = None
        self.distance_matrix = None
        self.activities = None
        self.linkage = None
        self.dendrogram = None
        self.clusterings = None
        self.hierarchies = None

    def get_log(self):
        return self.log

    def get_distances(self):
        return self.distances

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_activities(self):
        return self.activities

    def get_linkage(self):
        return self.linkage

    def get_dendrogram(self):
        return self.dendrogram

    def get_clusterings(self):
        return self.clusterings

    def get_hierarchies(self):
        return self.hierarchies
        
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

    def perform_clustering(self, verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using simple jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # generate a distance matrix using the directly follows graph of the events
        self.distances = self.get_jaccard_distance_of_dfg()
        self.activities, self.distance_matrix = self.convert_distances_to_distance_matrix(self.distances)
        
        # print the distances between events
        if verbose:
            print(self.distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix)
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)

        # generate the hierarchies dictionary
        self.clusterings = Clustering.create_clusterings_for_every_level(self.activities, self.distance_matrix, self.linkage)
        self.hierarchies = Clustering.create_hierarchy_for_activities(self.activities, self.clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(self.hierarchies)

class Weighted_Jaccard:
    def __init__(self, log):
        self.log = log
        self.connections_df = None
        self.distance_matrix = None
        self.activities = None
        self.linkage = None
        self.dendrogram = None
        self.clusterings = None
        self.hierarchies = None

    def get_log(self):
        return self.log

    def get_connections_df(self):
        return self.connections_df

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_activities(self):
        return self.activities

    def get_linkage(self):
        return self.linkage

    def get_dendrogram(self):
        return self.dendrogram

    def get_clusterings(self):
        return self.clusterings

    def get_hierarchies(self):
        return self.hierarchies

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

    def perform_clustering(self, verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using weighted jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''
        
        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # convert the log to a DataFrame with weights between the connections between events
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log)
        self.connections_df = Log_processing.get_weighted_df(self.connections_df)

        # generate the weighted jaccard distance matrix
        self.distance_matrix = self.get_weighted_jaccard_distance_matrix(self.connections_df)
        self.activities = self.distance_matrix.columns
        
        # print the distances between events
        if verbose:
            print(self.distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)
        
        # generate the hierarchies dictionary
        self.clusterings = Clustering.create_clusterings_for_every_level(self.activities, self.distance_matrix, self.linkage)
        self.hierarchies = Clustering.create_hierarchy_for_activities(self.activities, self.clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(self.hierarchies)

class Jaccard_N_grams:
    def __init__(self, log):
        self.log = log
        self.connections_df = None
        self.G = None
        self.distance_matrix = None
        self.activities = None
        self.linkage = None
        self.dendrogram = None
        self.clusterings = None
        self.hierarchies = None

    def get_log(self):
        return self.log

    def get_connections_df(self):
        return self.connections_df

    def get_G(self):
        return self.G

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_activities(self):
        return self.activities

    def get_linkage(self):
        return self.linkage

    def get_dendrogram(self):
        return self.dendrogram

    def get_clusterings(self):
        return self.clusterings

    def get_hierarchies(self):
        return self.hierarchies

    def get_predecessors_len_n(self, curr_node, length):
        """
        Recursively finds all variants of paths of length n that have an outgoing link at the last node to curr_node.
        In other words: If there is a path from a node A to curr_node of length n, then it is the predecessor_paths list
        
        Every solution always includes curr_node.
        Solution adapted from: https://stackoverflow.com/a/28103735
        """
        
        # Base case: Return a list with one list item curr_node
        if length == 0:
            return [[curr_node]]
        
        # the result list
        predecessor_paths = []
        
        # recursively generate predecessors_paths
        for neighbor in self.G.predecessors(curr_node):
            for path in self.get_predecessors_len_n(neighbor, length-1):
                # add curr_node only if it is not already in there, loops don't count
                if curr_node not in path:
                    predecessor_paths.append(path + [curr_node])
        
        return predecessor_paths

    def get_successors_len_n(self, curr_node, length):
        """
        Recursively finds all variants of paths of length n that have an ingoing link at the first node from curr_node.
        In other words: If there is a path from curr_node to a node A of length n, then it is the successor_paths list
        
        Every solution always includes curr_node.
        Solution adapted from: https://stackoverflow.com/a/28103735
        """

        # Base case: Return a list with one list item curr_node
        if length == 0:
            return [[curr_node]]
        
        # the result list
        successor_paths = []

        # recursively generate successors_path
        for neighbor in nx.descendants(self.G, curr_node):
            for path in self.get_successors_len_n(neighbor, length-1):
                # add curr_node only if it is not already in there, loops don't count
                if curr_node not in path:
                    successor_paths.append([curr_node] + path)

        return successor_paths

    def get_neighbours(self, start_node, length):
        """
        Get all paths of length n that lead to start_node and start from start_node and join them together
        """

        # get predecessor and descendant paths of start_node 
        paths_pred = self.get_predecessors_len_n(start_node, length)
        paths_succ = self.get_successors_len_n(start_node, length)

        # convert the nested list into list of tuples and remove the start_node
        paths_pred = [tuple(path[:-1]) for path in paths_pred]
        paths_succ = [tuple(path[1:]) for path in paths_succ]

        # convert to sets
        paths_pred = set(paths_pred)
        paths_succ = set(paths_succ)

        # join predecessors and descendants and remove duplicates
        paths = set(paths_pred).union(set(paths_succ))

        return paths

    def get_jaccard_distance(self, node1, node2, length):
        """
        Determine the Jaccard similarity between the two nodes.
        """

        # get neighbours of node1 and node2
        node1_neighbours = self.get_neighbours(node1, length)
        node2_neighbours = self.get_neighbours(node2, length)

        # convert the lists to sets
        node1_neighbours = set(node1_neighbours)
        node2_neighbours = set(node2_neighbours)

        # find the intersection and union of the sets
        intersection = node1_neighbours.intersection(node2_neighbours)
        union = node1_neighbours.union(node2_neighbours)

        return len(intersection) / len(union)

    def get_jaccard_distance_matrix(self, activities, length):
        """
        Computes the Jaccard distance between all pairs of labels and converts them to a distance matrix
        """

        jaccard_table = {} # the distances between the nodes

        # calculate the jaccard distances between all pairs of nodes and save them in jaccard_table
        for node1 in activities:
            node1_distances = {}
            for node2 in activities:
                similarity = self.get_jaccard_distance(node1, node2, length)
                node1_distances[node2] = round(1 - similarity, 6)
            jaccard_table[node1] = node1_distances

        jaccard_table = pd.DataFrame(jaccard_table)
        
        return jaccard_table
    
    def perform_clustering(self, length, verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using Jaccard with n_grams
        and then generating a hierarchy used for abstracting the event log
        '''

        # convert event log to an undirected NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(edge_list)

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # convert the log to a DataFrame 
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log)
        self.activities = self.connections_df.columns

        # generate the distance matrix
        self.distance_matrix = self.get_jaccard_distance_matrix(self.activities, length)

        # print the distances between events
        if verbose:
            print(self.distance_matrix)        
            
        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)
