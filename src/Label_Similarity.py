import pandas as pd
import pm4py
import spacy

import Log_processing, Clustering

class Label_Similarity:
    def __init__(self, log):
        self.log = log
        self.nlp = spacy.load("en_core_web_lg")
        self.distance_matrix = None
        self.activities = None
        self.linkage = None
        self.dendrogram = None
        self.clusterings = None
        self.hierarchies = None

    def calculate_label_similarity(self, act1_label, act2_label):
        """
        Computes the similarity of the label names using the Scipy NLP library. 
        Used pipeline package is en_core_web_lg, which is a bigger one to get more reliable results
        """

        # process the activity labels using scipy
        processed_act1 = self.nlp(act1_label)
        processed_act2 = self.nlp(act2_label)

        # get the similarity between two labels
        label_similarity = processed_act1.similarity(processed_act2)

        return label_similarity

    def generate_distance_matrix(self):
        """
        Computes the similarities between all pairs of labels and converts them to a distance matrix
        """

        distances = {} # the distances between the nodes

        # calculate the similarity between all pairs of nodes and save them in distances
        for act1_label in self.activities:
            act1_distances = {}
            for act2_label in self.activities:
                similarity = self.calculate_label_similarity(act1_label, act2_label)
                act1_distances[act2_label] = round(1 - similarity, 6)
            distances[act1_label] = act1_distances

        distances = pd.DataFrame(distances)

        return distances

    def perform_clustering(self, activity_key="concept:name", verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using label similarity
        and then generating a hierarchy used for abstracting the event log
        '''

        # display the dfg of the given log
        if verbose:
            Log_processing.show_dfg_of_log(self.log, activity_key=activity_key)

        # get all activity labels from the event log
        self.activities = pm4py.get_event_attribute_values(self.log, activity_key)
        self.activities = list(self.activities.keys())
        
        # generate the distance matrix between all pairs of labels
        self.distance_matrix = self.generate_distance_matrix()

        # print the distance matrix if necessary
        if verbose:
            print(self.distance_matrix)

        # perform hierarchical clustering and generate the resulting dendrogram
        self.linkage = Clustering.create_linkage(self.distance_matrix.to_numpy())
        self.dendrogram = Clustering.create_dendrogram(self.activities, self.linkage, ax=ax, no_plot=no_plot)

        # generate the hierarchies dictionary
        self.clusterings = Clustering.create_clusterings_for_every_level(self.activities, self.distance_matrix, self.linkage)
        self.hierarchies = Clustering.create_hierarchy_for_activities(self.activities, self.clusterings)