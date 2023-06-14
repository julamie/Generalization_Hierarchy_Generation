import pandas as pd
import pm4py
import networkx as nx

def get_log(path: str):
    return pm4py.read_xes(path)

def print_log_info(log):
    start_activities = pm4py.stats.get_start_activities(log)
    end_activities = pm4py.stats.get_end_activities(log)
    event_attributes = pm4py.stats.get_event_attributes(log)
    trace_attributes = pm4py.stats.get_trace_attributes(log)
#    event_attribute_values = [
#        {attribute: pm4py.stats.get_event_attribute_values(log, attribute)}
#        for attribute in event_attributes]
#    trace_attribute_values = [
#        {attribute: pm4py.stats.get_trace_attribute_values(log, attribute)}
#        for attribute in trace_attributes
#    ]
#    variants = pm4py.stats.get_variants(log)

    info = f"Event log information:\n"
    info += f"Start activities: {start_activities}\n"
    info += f"End activities: {end_activities}\n"
    info += f"Event attributes: {event_attributes}\n"
    info += f"Trace attributes: {trace_attributes}\n"
#    info += f"Event attribute values: {event_attribute_values}\n"
#    info += f"Trace attribute values: {trace_attribute_values}\n"
#    info += f"Variants: {variants}"
    print(info)

def save_dfg_of_log(log):
    dfg, start_activities, end_activities = pm4py.discovery.discover_dfg(log, 
                                                                         case_id_key= "case:concept:name",
                                                                         activity_key= "concept:name")
    pm4py.vis.save_vis_dfg(dfg, start_activities, end_activities, file_path="../out/dfg/sepsis.svg")

def get_jaccard_distance_of_dfg(log):
    # convert event log to an undirected NetworkX graph
    #dfg = pm4py.convert.convert_log_to_networkx(log, case_id_key="concept:name", include_df = True).to_undirected()
    dfg, start, end = pm4py.discover_dfg(log)
    edge_list = [(edge[0], edge[1], weight) for edge, weight in dfg.items()]
    dg = nx.DiGraph()
    dg.add_weighted_edges_from(edge_list)

    # compute the jaccard distances of the nodes
    distances = nx.jaccard_coefficient(dg.to_undirected())

    return distances

def convert_pairwise_distances_to_distance_matrix(log, pairwise_distances):
    activities = list(pm4py.get_event_attribute_values(log, "concept:name").keys())
    activities.sort()
    
    distance_matrix = pd.DataFrame(columns=activities, index=activities)
    
    for u, v, distance in pairwise_distances:
        distance_matrix[u][v] = 1 - round(distance, 3)
        distance_matrix[v][u] = 1 - round(distance, 3)
    distance_matrix = distance_matrix.fillna(1)
    
    return activities, distance_matrix.to_numpy()

if __name__ == "__main__":
    log = get_log("../logs/sepsis_event_log.xes")
    #print_log_info(log)
    pairwise_distances = get_jaccard_distance_of_dfg(log)
    activities, distances = convert_pairwise_distances_to_distance_matrix(log, pairwise_distances)
    #save_dfg_of_log(log)
    