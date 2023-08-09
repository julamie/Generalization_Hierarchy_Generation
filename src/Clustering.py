from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def create_linkage(distances):
    '''
    Performs hierarchical clustering using average linkage with scipy
    Returns the linkage matrix
    '''

    # input for the method has to be a condensed distance matrix
    condensed_matrix = squareform(distances)

    # perform hierarchical clustering
    return linkage(condensed_matrix, 'average')

def create_dendrogram(activities, linkage_matrix):
    '''
    Creates the dendrogram of the hierarchical clustering operation using scipy
    Displays the dendrogram to the terminal
    '''

    # create dendrogram out of data
    dn = dendrogram(linkage_matrix,
                    orientation="right",
                    labels=activities)

    return dn

def create_clusterings_for_every_level(activities, distances, linkage_matrix):
    '''
    Creates a dictionary for every clustering. Each entry includes a list of clusterings 
    for every level of the dendrogram.
    Returns the clustering dictionary
    '''

    # output dictionary and the number of done merges 
    clusterings = {}
    num_merges = 0

    # add a singleton cluster for every activity at first entry in clusterings dictionary
    dct = dict([(i, {activities[i]}) for i in range(distances.shape[0])])
    clusterings[0] = list(dct.values())

    # adapted from: https://stackoverflow.com/a/65060545
    for i, row in enumerate(linkage_matrix, distances.shape[0]):
        # add for every merge a union of the merged clusters and delete the old ones
        dct[i] = dct[row[0]].union(dct[row[1]])
        del dct[row[0]]
        del dct[row[1]]
        num_merges += 1

        # save the clustering in the output dictionary
        clusterings[num_merges] = list(dct.values())
    
    return clusterings

def create_hierarchy_for_activities(activities, clusterings):
    '''
    Returns every cluster each activity was ever in. This in return generates the hierarchy needed
    for abstraction of event logs
    Hint: Currect runtime is O(|activities|^2), there maybe is a faster method, though this is fine for now
    '''

    # create empty hierarchy lists for each activity
    hierarchies = {activity: [] for activity in activities}

    # for each cluster in each level add the new cluster to the corresponding activities
    for level in clusterings.values():
        for cluster in level:
            for activity in cluster:
                act_hierarchy = hierarchies[activity]
                # skip cluster if it is already in its hierarchy
                if cluster in act_hierarchy:
                    continue
                act_hierarchy.append(cluster)

    return hierarchies

def print_hierarchy_for_activities(hierarchies):
    print("------------------------------------------------------")
    print("The clusters each activity was in:")
    for activity, clusters in hierarchies.items():
        print(f"{activity}: ")
        for cluster in clusters:
            print(f"\t{cluster}")
    print("------------------------------------------------------")