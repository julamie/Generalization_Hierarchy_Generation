import Jaccard, Simrank, Role_Comparison, Label_Similarity
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import rand_score, mutual_info_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import tanglegram as tg
import pandas as pd
import pm4py
from pm4py.algo.organizational_mining.sna import algorithm as sna
import webbrowser

def show_jaccard_dendrograms_for_event_log(log, figure_title, output_file_name, activity_key="concept:name"):
    '''
    Prints the dendrograms from a given log using the Jaccard distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(figure_title)

    ax[0].set_title("Simple Jaccard")
    ax[1].set_title("Weighted Jaccard")

    simple_jaccard = Jaccard.Simple_Jaccard(log)
    weighted_jaccard = Jaccard.Weighted_Jaccard(log)

    simple_jaccard.perform_clustering(ax=ax[0], activity_key=activity_key)
    weighted_jaccard.perform_clustering(ax=ax[1], activity_key=activity_key)

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def show_simrank_dendrograms_for_event_log(log, figure_title, output_file_name, activity_key="concept:name"):
    '''
    Prints the dendrograms from a given log using the Simrank distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(figure_title)

    ax[0].set_title("Simple Simrank")
    ax[1].set_title("Weighted Simrank")

    simple_simrank = Simrank.Simple_Simrank(log)
    weighted_simrank = Simrank.Weighted_Simrank(log)

    simple_simrank.perform_clustering(ax=ax[0], activity_key=activity_key)
    weighted_simrank.perform_clustering(ax=ax[1], activity_key=activity_key)

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def show_n_gram_dendrograms_for_event_log(log, figure_title, output_file_name, activity_key="concept:name"):
    '''
    Prints the dendrograms from a given log using the N_gram Jaccard distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(4, 2, figsize=(15, 15))
    fig.suptitle(figure_title)

    ax[0, 0].set_title("Simple Jaccard N_gram length = 1")
    ax[0, 1].set_title("Weighted Jaccard N_gram length = 1")
    ax[1, 0].set_title("Simple Jaccard N_gram length = 2")
    ax[1, 1].set_title("Weighted Jaccard N_gram length = 2")
    ax[2, 0].set_title("Simple Jaccard N_gram length = 3")
    ax[2, 1].set_title("Weighted Jaccard N_gram length = 3")
    ax[3, 0].set_title("Simple Jaccard N_gram length = 4")
    ax[3, 1].set_title("Weighted Jaccard N_gram length = 4")

    simple_n_gram = Jaccard.Jaccard_N_grams(log)
    weighted_n_gram = Jaccard.Weighted_Jaccard_N_grams(log)

    simple_n_gram.perform_clustering(length=1, ax=ax[0, 0], activity_key=activity_key)
    weighted_n_gram.perform_clustering(length=1, ax=ax[0, 1], activity_key=activity_key)
    simple_n_gram.perform_clustering(length=2, ax=ax[1, 0], activity_key=activity_key)
    weighted_n_gram.perform_clustering(length=2, ax=ax[1, 1], activity_key=activity_key)
    simple_n_gram.perform_clustering(length=3, ax=ax[2, 0], activity_key=activity_key)
    weighted_n_gram.perform_clustering(length=3, ax=ax[2, 1], activity_key=activity_key)
    simple_n_gram.perform_clustering(length=4, ax=ax[3, 0], activity_key=activity_key)
    weighted_n_gram.perform_clustering(length=4, ax=ax[3, 1], activity_key=activity_key)

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def show_role_comparison_dendrograms_for_event_log(log, activities_column, roles_column, figure_title, output_file_name):
    '''
    Prints the dendrograms from a given log using the role comparison distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(figure_title)

    ax[0].set_title("Simple Role Comparison")
    ax[1].set_title("Weighted Role Comparison")

    role_comp = Role_Comparison.Role_Comparison(log, activities_column=activities_column, roles_column=roles_column)
    role_comp.perform_clustering(ax=ax[0], weighted=False)
    role_comp.perform_clustering(ax=ax[1], weighted=True)

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def show_label_similarity_dendrograms_for_event_log(log, figure_title, output_file_name, activity_key="concept:name"):
    '''
    Prints the dendrograms from a given log using the label similarity distance measures, saves the generated figure in out folder
    '''

    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(figure_title)

    role_comp = Label_Similarity.Label_Similarity(log)
    role_comp.perform_clustering(ax=ax, activity_key=activity_key)

    fig.tight_layout()
    plt.show()
    fig.savefig("../out/" + output_file_name)

def compare_dendrogram_using_rand_score(metric1, metric2):
    '''
    Calculates the rand score between two clusterings for every level. Plots the result
    '''

    # save their linkages
    metric1_linkage = metric1.linkage
    metric2_linkage = metric2.linkage

    # create a dictionary for every clustering in each level in the dendrogram
    metric1_dict = {}
    metric2_dict = {}

    # every level two clusters merge to one. There are number of activities
    # many clusters at the beginning
    number_of_levels = len(metric1.activities)

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

    # save their linkages
    metric1_linkage = metric1.linkage
    metric2_linkage = metric2.linkage

    # create a dictionary for every clustering in each level in the dendrogram
    metric1_dict = {}
    metric2_dict = {}

    # every level two clusters merge to one. There are number of activities
    # many clusters at the beginning
    number_of_levels = len(metric1.activities)

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

def compare_dendrogram_using_fowlkes_mallows_score(metric1, metric2):
    '''
    Calculates the Fowlkes-Mallows score between two clusterings for every level. Plots the result
    '''

    # save their linkages
    metric1_linkage = metric1.linkage
    metric2_linkage = metric2.linkage

    # create a dictionary for every clustering in each level in the dendrogram
    metric1_dict = {}
    metric2_dict = {}

    # every level two clusters merge to one. There are number of activities
    # many clusters at the beginning
    number_of_levels = len(metric1.activities)

    # add every clustering in each level to their dictionary
    for num_clusters in range(1, number_of_levels):
        metric1_dict[num_clusters] = fcluster(metric1_linkage, num_clusters, criterion='maxclust')
        metric2_dict[num_clusters] = fcluster(metric2_linkage, num_clusters, criterion='maxclust')
    
    # create list of mutual info score per level
    fm_scores = {}
    for num_clusters in range(1, number_of_levels):
        fm_scores[num_clusters] = fowlkes_mallows_score(metric1_dict[num_clusters], metric2_dict[num_clusters])

    # plot data
    fig = plt.figure()
    plt.title("Fowlkes-Mallows scores per number of clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Fowlkes-Mallows score")
    plt.plot(list(fm_scores.keys()), list(fm_scores.values()))
    fig.show()

    return fm_scores

def show_tanglegram(metric1, metric2):
    '''
    Shows a tanglegram of two used metrics to show which labels have been assigned differently
    '''

    metric1.perform_clustering(no_plot=True)
    metric2.perform_clustering(no_plot=True)

    jacc_df = pd.DataFrame(metric1.get_distance_matrix(), columns = metric1.get_activities(), index = metric1.get_activities())
    simrank_df = pd.DataFrame(metric2.get_distance_matrix(), columns = metric2.get_activities(), index = metric2.get_activities())

    fig = tg.plot(jacc_df, simrank_df, sort=True)
    fig.set_size_inches(32, 10)
    plt.show()

def save_and_open_handover_graph_of_log(log, resource_key, filename):
    '''
    Creates a handover graph of the given log with given attribute and saves it at ../out/{filename} as an html file.
    Opens this file in a browser
    '''

    handover = pm4py.discover_handover_of_work_network(log, resource_key=resource_key, timestamp_key='time:timestamp', case_id_key='case:concept:name')
    pm4py.save_vis_sna(handover, f"out/{filename}", variant_str=sna.Variants.HANDOVER_LOG)
    webbrowser.open_new_tab(f"out/{filename}")
