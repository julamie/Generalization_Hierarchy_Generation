from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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

def create_dendrogram(activities, linkage_matrix, ax=None, no_plot=False):
    '''
    Creates the dendrogram of the hierarchical clustering operation using scipy
    Displays the dendrogram to the terminal
    '''

    # create dendrogram out of data
    dn = dendrogram(linkage_matrix,
                    orientation="right",
                    labels=activities,
                    ax=ax,
                    no_plot=no_plot)

    return dn

def get_clustering_for_n_clusters(linkage, activities, num_clusters):
    '''
    Returns a list of all clusters for the given number of clusters needed
    '''

    # handle edge cases
    if num_clusters == len(activities):
        # every activity has its own cluster
        return [[activity] for activity in activities]
    elif num_clusters == 1:
        # if only one cluster is needed, then all activities are there
        return [list(activities)]

    # use scipy fcluster to get the cluster of every activity
    cluster_list = fcluster(linkage, num_clusters, criterion='maxclust')

    # stores the activities in the same cluster
    cluster_data = [[] for _ in range(num_clusters)]
    
    # append the activity name in the corresponding cluster list
    for i, event in enumerate(cluster_list):
        cluster_data[event - 1].append(activities[i])

    return cluster_data

def create_list_num_clusters_per_level(num_activities, num_levels):
    '''
    Creates a list. Each element states how many clusters should be formed for every level.
    The number of clusters is evenly divided.
    At level 0 there should exist as many clusters as there are activities and
    at the last level should be only one cluster with all activities.
    '''

    num_clusters_per_level = []
    subtrahend = num_activities / (num_levels - 1)
    num_clusters = num_activities

    # num_clusters keeps getting smaller in each iteration by subtrahend
    while int(num_clusters) > 1:
        num_clusters_per_level.append(int(num_clusters))
        num_clusters -= subtrahend
    num_clusters_per_level.append(1)

    return num_clusters_per_level

def create_hierarchy_levels(activities, cluster_data):
    '''
    Creates a dictionary where every activity is a key. Each item is the hierarchy of clusters,
    for the number of clusters given before. Each hierarchy has the same length.
    '''
    
    hierarchies = {activity: [] for activity in activities}

    # iterate over every clustering for each number of clusters
    for curr_clustering in cluster_data:
        # for every activity in curr_cluster, add curr_cluster to the corresponding activity list
        for curr_cluster in curr_clustering:
            for activity in curr_cluster:
                hierarchies[activity].append(curr_cluster)

    return hierarchies

def generate_hierarchy_file(linkage, activities, num_levels, file_name):
    '''
    Generates the file for the hierarchy of activities in out/file_name.
    Every row is the hierarchy for one activity. The first cluster is unchanged,
    the following clusters are surrounded by curly brackets, the last cluster is
    the *, which includes all activities
    '''
    
    # get the number of clusters needed for each level in the hierarchy
    num_clusters_per_level = create_list_num_clusters_per_level(len(activities), num_levels)

    # generate the cluster_data using num_clusters_per_level
    cluster_data = []
    for num_clusters in num_clusters_per_level:
        curr_clustering = get_clustering_for_n_clusters(linkage, activities, num_clusters)
        cluster_data.append(curr_clustering)   
    
    # convert cluster_data to the hierarchies dictionary
    hierarchies = create_hierarchy_levels(activities, cluster_data)

    # write the hierarchy output file
    with open(f"../out/{file_name}", 'w') as f:
        for activity in hierarchies:
            first_element = True
            for level in hierarchies[activity]:
                # write the fast element without curly brackets
                if first_element:
                    f.write(str(level[0]) + ";")
                    first_element = False
                # if all activities are in the cluster, shorten output to *
                elif len(level) == len(activities):
                    f.write("*\n")
                # surround the content of the cluster by curly brackets
                else:
                    cluster_to_string = "{" + ', '.join(level) + "};"
                    f.write(cluster_to_string)

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