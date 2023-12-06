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

    def get_neighbours(self, curr_node):
        """
        Get all predecessors and successors of curr_node and join them together
        """

        # get predecessors and successors
        preds = self.G.predecessors(curr_node)
        succs = self.G.successors(curr_node)

        # convert to set for union operation
        preds = set(preds)
        succs = set(succs)

        # neighbours are the union of those sets
        neighbours = preds.union(succs)

        return neighbours

    def get_jaccard_distance(self, node1, node2, split_neighbours=False):
        """
        Determine the Jaccard similarity between the two nodes.
        """

        # skip evaluation step if the nodes are the same
        if node1 == node2:
            return 1
        
        # if split_neighbours is true, we calculate the Jaccard for predecessors and
        # successors separately and use the average
        if split_neighbours:
            node1_preds = set(self.G.predecessors(node1))
            node2_preds = set(self.G.predecessors(node2))

            if len(node1_preds) == 0 or len(node2_preds) == 0:
                return 0
            
            node1_preds = set(node1_preds)
            node2_preds = set(node2_preds)

            intersection_pred = node1_preds.intersection(node2_preds)
            union_pred = node1_preds.union(node2_preds)

            jacc_pred = len(intersection_pred) / len(union_pred)

            # ------

            node1_succ = set(self.G.successors(node1))
            node2_succ = set(self.G.successors(node2))

            if len(node1_succ) == 0 or len(node2_succ) == 0:
                return 0
            
            node1_succ = set(node1_succ)
            node2_succ = set(node2_succ)

            intersection_succ = node1_succ.intersection(node2_succ)
            union_succ = node1_succ.union(node2_succ)

            succ_pred = len(intersection_succ) / len(union_succ)
            # ------

            return (jacc_pred + succ_pred) / 2
        
        # if split_neighbours is False, the combine predecessors and successors
        else:
            # get neighbours of node1 and node2
            node1_neighbours = self.get_neighbours(node1)
            node2_neighbours = self.get_neighbours(node2)

            # handle edge cases where there were no paths of length n found
            if len(node1_neighbours) == 0 or len(node2_neighbours) == 0:
                return 0 

            # convert the lists to sets
            node1_neighbours = set(node1_neighbours)
            node2_neighbours = set(node2_neighbours)

            # find the intersection and union of the sets
            intersection = node1_neighbours.intersection(node2_neighbours)
            union = node1_neighbours.union(node2_neighbours)

            return len(intersection) / len(union)

    def get_jaccard_distance_matrix(self, activities):
        """
        Computes the Jaccard distance between all pairs of labels and converts them to a distance matrix
        """

        jaccard_table = {} # the distances between the nodes

        # calculate the jaccard distances between all pairs of nodes and save them in jaccard_table
        for node1 in activities:
            node1_distances = {}
            for node2 in activities:
                similarity = self.get_jaccard_distance(node1, node2)
                node1_distances[node2] = round(1 - similarity, 6)
            jaccard_table[node1] = node1_distances

        jaccard_table = pd.DataFrame(jaccard_table)
        
        return jaccard_table

    def perform_clustering(self, activity_key="concept:name", verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using simple jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''
        # convert event log to an undirected NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log, activity_key=activity_key)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(edge_list)

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log, activity_key=activity_key)

        # convert the log to a DataFrame 
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log, activity_key=activity_key)
        self.activities = self.connections_df.columns

        # generate the distance matrix
        self.distance_matrix = self.get_jaccard_distance_matrix(self.activities)

        # print the distances between events
        if verbose:
            print(self.distance_matrix)        
            
        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)

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
            
            # check the other direction and change the weight to the bigger one
            if node1 in G[x]:
                if G[x][node1]['weight'] > node1_weight:
                    node1_weight = G[x][node1]['weight']
            if node2 in G[x]:
                if G[x][node2]['weight'] > node2_weight:
                    node2_weight = G[x][node2]['weight']
            

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

    def perform_clustering(self, activity_key="concept:name", verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using weighted jaccard similarity
        and then generating a hierarchy used for abstracting the event log
        '''
        
        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log, activity_key=activity_key)

        # convert the log to a DataFrame with weights between the connections between events
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log, activity_key=activity_key)
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
        for neighbor in self.G.successors(curr_node):
            for path in self.get_successors_len_n(neighbor, length-1):
                successor_paths.append([curr_node] + path)

        return successor_paths

    def get_neighbours(self, curr_node, length):
        """
        Get all paths of length n that lead to start_node and start from start_node and join them together
        """

        # get predecessor and descendant paths of start_node 
        paths_pred = self.get_predecessors_len_n(curr_node, length)
        paths_succ = self.get_successors_len_n(curr_node, length)

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

        # skip evaluation step if the nodes are the same
        if node1 == node2:
            return 1

        node1_neighbours = []
        node2_neighbours = []

        # get the neighbour paths of all lengths less than the given length and add them to nodeX_neighbours
        while length >= 1:
            # get neighbours of node1 and node2
            node1_neighbours += self.get_neighbours(node1, length)
            node2_neighbours += self.get_neighbours(node2, length)

            length -= 1

        # convert the lists to sets
        node1_neighbours = set(node1_neighbours)
        node2_neighbours = set(node2_neighbours)

        # handle edge cases where there were no paths of length n found
        if len(node1_neighbours) == 0 or len(node2_neighbours) == 0:
            return 0 

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
    
    def perform_clustering(self, length, activity_key="concept:name", verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using Jaccard with n_grams
        and then generating a hierarchy used for abstracting the event log
        '''

        # convert event log to an undirected NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log, activity_key=activity_key)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(edge_list)

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log, activity_key=activity_key)

        # convert the log to a DataFrame 
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log, activity_key=activity_key)
        self.activities = self.connections_df.columns

        # generate the distance matrix
        self.distance_matrix = self.get_jaccard_distance_matrix(self.activities, length)

        # print the distances between events
        if verbose:
            print(self.distance_matrix)        
            
        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)

class Weighted_Jaccard_N_grams:
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
        for neighbor in self.G.successors(curr_node):
            for path in self.get_successors_len_n(neighbor, length-1):
                successor_paths.append([curr_node] + path)

        return successor_paths

    def get_neighbours(self, curr_node, length):
        """
        Get all paths of length n that lead to start_node and start from start_node and join them together
        """

        # get predecessor and descendant paths of start_node 
        paths_pred = self.get_predecessors_len_n(curr_node, length)
        paths_succ = self.get_successors_len_n(curr_node, length)

        # convert the nested list into list of tuples and remove the start_node
        paths_pred = [tuple(path[:-1]) for path in paths_pred]
        paths_succ = [tuple(path[1:]) for path in paths_succ]

        # convert to sets
        paths_pred = set(paths_pred)
        paths_succ = set(paths_succ)

        # join predecessors and descendants and remove duplicates
        paths = set(paths_pred).union(set(paths_succ))

        return paths

    def get_weighted_jaccard_distance(self, node1, node2, length):
        """
        Determine the Weighted Jaccard similarity between the two nodes. Multiplies the weight of the nodes in a path
        and then calculates the similarity similar to Weighted_Jaccard's similarity calculation method
        """

        # skip evaluation step if the nodes are the same
        if node1 == node2:
            return 1

        node1_neighbours = []
        node2_neighbours = []
        
        # get the neighbour paths of all lengths less than the given length and add them to nodeX_neighbours
        while length >= 1:
            # get neighbours of node1 and node2
            node1_neighbours += self.get_neighbours(node1, length)
            node2_neighbours += self.get_neighbours(node2, length)

            length -= 1

        # convert the lists to sets
        node1_neighbours = set(node1_neighbours)
        node2_neighbours = set(node2_neighbours)

        # handle edge cases where there were no paths of length n found
        if len(node1_neighbours) == 0 or len(node2_neighbours) == 0:
            return 0 

        # find the intersection of the sets
        intersection = node1_neighbours.intersection(node2_neighbours)

        minimumns = 0
        maximums = 0

        # multiply every connection in a path and use this to calculate the weighted Jaccard similarity between the nodes
        for path in intersection:
            node1_path_weight = 1
            node2_path_weight = 1

            # start nodes
            node1_from_node = node1
            node2_from_node = node2
            for to_node in path:
                # multiply the edge weight to nodeX_path_weight
                node1_path_weight *= self.connections_df[to_node][node1_from_node]
                node2_path_weight *= self.connections_df[to_node][node2_from_node]
               
                # change the start node in path to get right edge weight in next iteration
                node1_from_node = to_node
                node2_from_node = to_node

            # sum up all the weights with minimal and maximal value
            minimumns += min(node1_path_weight, node2_path_weight)
            maximums += max(node1_path_weight, node2_path_weight)
        
        # prevent division by 0 error
        if maximums == 0:
            return 0

        return minimumns / maximums

    def get_weighted_jaccard_distance_matrix(self, activities, length):
        """
        Computes the Weighted Jaccard distance between all pairs of labels and converts them to a distance matrix
        """

        jaccard_table = {} # the distances between the nodes

        # calculate the weighted jaccard distances between all pairs of nodes and save them in jaccard_table
        for node1 in activities:
            node1_distances = {}
            for node2 in activities:
                similarity = self.get_weighted_jaccard_distance(node1, node2, length)
                node1_distances[node2] = round(1 - similarity, 6)
            jaccard_table[node1] = node1_distances

        jaccard_table = pd.DataFrame(jaccard_table)
        
        return jaccard_table

    def perform_clustering(self, length, activity_key="concept:name", verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using Weighted Jaccard with n_grams
        and then generating a hierarchy used for abstracting the event log
        '''

        # convert event log to an undirected NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log, activity_key=activity_key)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(edge_list)
        
        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log, activity_key=activity_key)

        # convert the log to a DataFrame 
        self.connections_df = Log_processing.get_pivot_df_from_dfg(self.log, activity_key=activity_key)
        self.connections_df = Log_processing.get_weighted_df(self.connections_df)
        self.activities = self.connections_df.columns
        
        # generate the distance matrix
        self.distance_matrix = self.get_weighted_jaccard_distance_matrix(self.activities, length)
        
        # print the distances between events
        if verbose:
            print(self.distance_matrix)        
            
        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)
        