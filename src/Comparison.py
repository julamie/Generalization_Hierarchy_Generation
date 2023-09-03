import Jaccard, Simrank
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import rand_score, mutual_info_score
import matplotlib.pyplot as plt
import tanglegram as tg
import pandas as pd

def show_dendrograms_for_event_log(log, figure_title, output_file_name):
    '''
    Prints the dendrograms from a given log using all distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(figure_title)

    ax[0, 0].set_title("Simple Jaccard")
    ax[0, 1].set_title("Weighted Jaccard")
    ax[1, 0].set_title("Simple Simrank")
    ax[1, 1].set_title("Weighted Simrank")

    simple_jaccard = Jaccard.Simple_Jaccard(log)
    weighted_jaccard = Jaccard.Weighted_Jaccard(log)
    simple_simrank = Simrank.Simple_Simrank(log)
    weighted_simrank = Simrank.Weighted_Simrank(log)

    simple_jaccard.perform_clustering(ax=ax[0, 0])
    weighted_jaccard.perform_clustering(ax=ax[0, 1])
    simple_simrank.perform_clustering(ax=ax[1, 0])
    weighted_simrank.perform_clustering(ax=ax[1, 1])

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def compare_dendrogram_using_rand_score(metric1, metric2):
    '''
    Calculates the rand score between two clusterings for every level. Plots the result
    '''

    # perform clustering using the different distance metrics given
    metric1.perform_clustering(no_plot=True)
    metric2.perform_clustering(no_plot=True)

    # save their linkages
    metric1_linkage = metric1.get_linkage()
    metric2_linkage = metric2.get_linkage()

    # create a dictionary for every clustering in each level in the dendrogram
    metric1_dict = {}
    metric2_dict = {}

    # every level two clusters merge to one. There are number of activities
    # many clusters at the beginning
    number_of_levels = len(metric1.get_activities())

    # add every clustering in each level to their dictionary
    for num_clusters in range(1, number_of_levels):
        metric1_dict[num_clusters] = fcluster(metric1_linkage, num_clusters, criterion='maxclust')
        metric2_dict[num_clusters] = fcluster(metric2_linkage, num_clusters, criterion='maxclust')
    
    # create list of rand scores per level
    rand_scores = {}
    for num_clusters in range(1, number_of_levels):
        rand_scores[num_clusters] = rand_score(metric1_dict[num_clusters], metric2_dict[num_clusters])

    # plot data
    fig = plt.figure()
    plt.title("Rand scores per number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Rand score between two metrics")
    plt.plot(list(rand_scores.keys()), list(rand_scores.values()))
    fig.show()

    return rand_scores

def compare_dendrogram_using_mutual_info_score(metric1, metric2):
    '''
    Calculates the mutual information score between two clusterings for every level. Plots the result
    '''

    # perform clustering using the different distance metrics given
    metric1.perform_clustering(no_plot=True)
    metric2.perform_clustering(no_plot=True)

    # save their linkages
    metric1_linkage = metric1.get_linkage()
    metric2_linkage = metric2.get_linkage()

    # create a dictionary for every clustering in each level in the dendrogram
    metric1_dict = {}
    metric2_dict = {}

    # every level two clusters merge to one. There are number of activities
    # many clusters at the beginning
    number_of_levels = len(metric1.get_activities())

    # add every clustering in each level to their dictionary
    for num_clusters in range(1, number_of_levels):
        metric1_dict[num_clusters] = fcluster(metric1_linkage, num_clusters, criterion='maxclust')
        metric2_dict[num_clusters] = fcluster(metric2_linkage, num_clusters, criterion='maxclust')
    
    # create list of mutual info score per level
    mutual_info_scores = {}
    for num_clusters in range(1, number_of_levels):
        mutual_info_scores[num_clusters] = mutual_info_score(metric1_dict[num_clusters], metric2_dict[num_clusters])

    # plot data
    fig = plt.figure()
    plt.title("Mutual info scores per number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Mutual info score between two metrics")
    plt.plot(list(mutual_info_scores.keys()), list(mutual_info_scores.values()))
    fig.show()

    return mutual_info_scores

def show_tanglegram(metric1, metric2):
    metric1.perform_clustering(no_plot=True)
    metric2.perform_clustering(no_plot=True)

    jacc_df = pd.DataFrame(metric1.get_distance_matrix(), columns = metric1.get_activities(), index = metric1.get_activities())
    simrank_df = pd.DataFrame(metric2.get_distance_matrix(), columns = metric2.get_activities(), index = metric2.get_activities())

    fig = tg.plot(jacc_df, simrank_df, sort=True)
    fig.set_size_inches(32, 10)
    plt.show()