import pandas as pd

import Clustering

class Role_Comparison:
    def __init__(self, log, activities_column, roles_column):
        self.log = log
        self.role_counts = None
        self.distance_matrix = None
        self.activities = None
        self.roles = None
        self.activities_column = activities_column
        self.roles_column = roles_column
        self.linkage = None
        self.dendrogram = None
        self.clusterings = None
        self.hierarchies = None

    def get_log(self):
        return self.log

    def get_role_counts(self):
        return self.role_counts

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_activities(self):
        return self.activities

    def get_roles(self):
        return self.roles

    def get_activities_column(self):
        return self.activities_column

    def get_roles_column(self):
        return self.roles_column

    def get_linkage(self):
        return self.linkage

    def get_dendrogram(self):
        return self.dendrogram

    def get_clusterings(self):
        return self.clusterings

    def get_hierarchies(self):
        return self.hierarchies
    
    def generate_role_counts(self):
        """
        Generates a DataFrame for every pair of activity and role. If the role doesn't perform the activity, the value is 0.
        If the role performs e.g. 70% of all executions of an activity, its value will be 0.7 
        """

        # Group the log by the activities and the roles, count how many times a roles has been executed per activity
        self.role_counts = self.log.groupby([self.activities_column, self.roles_column])[self.roles_column].count()

        # Divide the role counts by the number of different roles performing the activity
        self.role_counts = self.role_counts / self.log.groupby(self.activities_column).size()
        self.role_counts = self.role_counts.reset_index()

        # save the activities and roles
        self.activities = self.role_counts[self.activities_column].unique()
        self.roles = self.role_counts[self.roles_column].unique()

        # make a pivot table using the values and replace the NaNs by 1
        self.role_counts = self.role_counts.pivot(index=self.activities_column, columns=self.roles_column, values=0)
        self.role_counts = self.role_counts.fillna(1)
        return self.role_counts

    def calculate_jaccard_similarity(self, act1, act2, weighted):
        """
        Calculates the jaccard similarity of act1 and act2. If they share many roles performing them, the jaccard similarity is high.
        """

        # skip evaluation step if the nodes are the same
        if act1 == act2:
            return 1

        # get the role counts for act1 and act2
        act1_roles = self.role_counts.loc[act1]
        act2_roles = self.role_counts.loc[act2]

        minimums = 0
        maximums = 0
        
        # check every role
        for role in self.roles:
            # get the share how much out of all roles this particular role executes the activity
            if weighted:
                act1_weight = act1_roles[role]
                act2_weight = act2_roles[role]
            else:
                # if simple jaccard should be used, the weight is 1 if there is a weight greater than 0
                if act1_roles[role] > 0:
                    act1_weight = 1
                else:
                    act1_weight = 0

                if act2_roles[role] > 0:
                    act2_weight = 1
                else:
                    act2_weight = 0

            # sum up all the weights with minimal and maximal value
            minimums += min(act1_weight, act2_weight)
            maximums += max(act1_weight, act2_weight)

        # prevent division by 0 error
        if maximums == 0:
            return 0
        
        return minimums / maximums

    def get_distance_matrix(self, weighted):
        """
        Computes the (Weighted) Jaccard distance between all pairs of activities. The more roles two activities share, the lower the distance
        between them. All distances are converted to a DataFrame.
        """

        # saves all distances between activities
        jaccard_table = {}

        # calculate the weighted jaccard distances between all pairs of nodes and save them in jaccard_table
        for act1 in self.activities:
            act1_distances = {}
            for act2 in self.activities:
                similarity = self.calculate_jaccard_similarity(act1, act2, weighted=weighted)
                act1_distances[act2] = round(1 - similarity, 6)
            jaccard_table[act1] = act1_distances

        jaccard_table = pd.DataFrame(jaccard_table)
        return jaccard_table

    def perform_clustering(self, weighted=True, verbose=False, ax=None, no_plot=False):
        '''
        Performs all necessary steps to perform hierarchical clustering using the Jaccard of same roles performed
        and then generating a hierarchy used for abstracting the event log
        '''
        
        # generates the DataFrame which role performs which activity 
        self.generate_role_counts()

        # display the role_counts Series if neccessary
        if verbose:
            print(self.role_counts)

        # generate the distance matrix using the weighted Jaccard of same roles performed 
        self.distance_matrix = self.get_distance_matrix(weighted=weighted)
        
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