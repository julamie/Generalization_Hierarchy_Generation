import pandas as pd
import pm4py
import networkx as nx
import numpy as np
import Log_processing, Clustering

import SimRank_external # imported from https://github.com/ysong1231/SimRank/tree/main

class Simple_Simrank:
    def __init__(self, log_path, filtered=False, num_top_k=0):
        if filtered:
            if num_top_k == 0:
                print("num_top_k = 0 returns an empty event log")
            self.log = Log_processing.get_filtered_log(log_path, num_top_k)
        else:
            self.log = Log_processing.get_log(log_path)

    def get_simrank_distance_of_dfg(self):
        '''
        Computes the simrank similarity of two nodes using the DFG

        Output is a dictionary of dictionaries with the first key being the source node, the key
        in the second dictionary being the target node, and the value being the simrank similarity.
        The distance relation between source and target are symmetrical. 
        '''

        # convert event log to a NetworkX graph
        dfg, start, end = pm4py.discover_dfg(self.log)
        edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
        dg = nx.DiGraph()
        dg.add_weighted_edges_from(edge_list)

        # compute the simrank similarity of the nodes
        distances = nx.simrank_similarity(dg)

        return distances

    def convert_distances_to_distance_matrix(self, pairwise_distances):
        '''
        Converts the pairwise distances dictionary of dictionaries to a distance matrix
        '''

        # create distance matrix using the pairwise distances dictionary and sort them
        distance_matrix = pd.DataFrame(pairwise_distances)
        distance_matrix = distance_matrix.reindex(sorted(distance_matrix.columns), axis="index")
        distance_matrix = distance_matrix.reindex(sorted(distance_matrix.columns), axis="columns")
        activities = distance_matrix.columns

        # transform values and convert to numpy matrix
        distance_matrix = distance_matrix.transform(lambda x: round(1 - x, 6))
        distance_matrix = distance_matrix.to_numpy()
        
        # return only the values of the matrix, not the names of the activities
        return activities, distance_matrix

    def perform_clustering(self, verbose=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using simple simrank similarity
        and then generates a hierarchy used for abstracting the event log
        '''

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)
        
        # generate a distance matrix using the directly follows graph of the events
        pairwise_distances = self.get_simrank_distance_of_dfg()
        activities, distance_matrix = self.convert_distances_to_distance_matrix(pairwise_distances)
        
        # print the distances between events
        if verbose:
            display(distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        linkage = Clustering.create_linkage(distance_matrix)
        dendrogram = Clustering.create_dendrogram(activities, linkage)

        # generate the hierarchies dictionary
        clusterings = Clustering.create_clusterings_for_every_level(activities, distance_matrix, linkage)
        hierarchies = Clustering.create_hierarchy_for_activities(activities, clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(hierarchies)

class Weighted_Simrank:
    def __init__(self, log_path):
        self.log = Log_processing.get_log(log_path)

    def perform_clustering(self, verbose=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using weighted simrank similarity
        and then generates a hierarchy used for abstracting the event log
        '''

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log)

        # convert the log to a DataFrame with weights between the connections between events
        connections_df = Log_processing.get_df_from_dfg(self.log)

        # use external Simrank file to compute the weighted simrank distances
        sr = SimRank_external.SimRank()
        distance_matrix = sr.fit(data=connections_df,
                            verbose=False,
                            weighted = True,
                            from_node_column="From",
                            to_node_column="To",
                            weight_column="Frequency")
        distance_matrix = distance_matrix.transform(lambda x: 1 - x)
        activities = distance_matrix.columns

        # print the distances between events
        if verbose:
            display(distance_matrix)

        # apply min-max normalization to the entire array
        # otherwise the dendrogram looks unreadable
        distance_matrix = distance_matrix.to_numpy()
        np.fill_diagonal(distance_matrix, np.nan)
        min_val = np.nanmin(distance_matrix)
        max_val = np.nanmax(distance_matrix)
        normalized_data = np.vectorize(lambda x: (x - min_val) / (max_val - min_val))(distance_matrix)
        np.fill_diagonal(normalized_data, 0)
        distance_matrix = normalized_data

        # perform hierarchical clustering and generate the resulting dendrogram
        linkage = Clustering.create_linkage(distance_matrix)
        dendrogram = Clustering.create_dendrogram(activities, linkage)

        # generate the hierarchies dictionary
        clusterings = Clustering.create_clusterings_for_every_level(activities, distance_matrix, linkage)
        hierarchies = Clustering.create_hierarchy_for_activities(activities, clusterings)

        # print the hierarchies
        if verbose:
            Clustering.print_hierarchy_for_activities(hierarchies)