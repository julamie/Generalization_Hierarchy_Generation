import pandas as pd
import pm4py

def get_log(path):
    '''
    Reads in an event log from the logs folder using pm4py 
    '''

    return pm4py.read_xes(path)

def get_filtered_log(path, num_top_k):
    filtered_log = get_log(path)
    filtered_log = pm4py.filter_variants_top_k(filtered_log, num_top_k)

    return filtered_log

def print_log_info(log, verbose=False):
    '''
    Prints out useful information of the given event log

    Prints out all start activities, end activities, all event attributes, and all trace attributes.
    If verbose is True, then all values of all event and trace attributes are printed out plus all variants of the event log
    '''

    start_activities = pm4py.stats.get_start_activities(log)
    end_activities = pm4py.stats.get_end_activities(log)
    event_attributes = pm4py.stats.get_event_attributes(log)
    trace_attributes = pm4py.stats.get_trace_attributes(log)
    
    info = f"Event log information:\n"
    info += f"Start activities: {start_activities}\n"
    info += f"End activities: {end_activities}\n"
    info += f"Event attributes: {event_attributes}\n"
    info += f"Trace attributes: {trace_attributes}\n"

    # add additional info if wished for
    if verbose:
        event_attribute_values = [
            {attribute: pm4py.stats.get_event_attribute_values(log, attribute)}
            for attribute in event_attributes
        ]
        trace_attribute_values = [
            {attribute: pm4py.stats.get_trace_attribute_values(log, attribute)}
            for attribute in trace_attributes
        ]
        variants = pm4py.stats.get_variants(log)

        info += f"Event attribute values: {event_attribute_values}\n"
        info += f"Trace attribute values: {trace_attribute_values}\n"
        info += f"Variants: {variants}"

    print(info)

def show_dfg_of_log(log, activity_key="concept:name"):
    '''
    Displays the directly follows graph of an event log
    '''

    dfg, start_activities, end_activities = pm4py.discovery.discover_dfg(log, 
                                                                         case_id_key= "case:concept:name",
                                                                         activity_key= activity_key)

    return pm4py.vis.view_dfg(dfg, start_activities, end_activities, format="png")

def get_df_from_dfg(log, activity_key="concept:name"):
    '''
    Reads in the log and then produces a dfg from it. The function then generates a Pandas DataFrame.

    Every row in the DataFrame has a source node, a target node and the frequency of the vertex
    '''

    dfg, _, _ = pm4py.discover_dfg(log, activity_key=activity_key)
    connections_list = [[edges[0], edges[1], weight] for edges, weight in dfg.items()] 
    df = pd.DataFrame(data=connections_list, columns=["From", "To", "Frequency"])

    return df

def get_pivot_df_from_dfg(log, activity_key="concept:name"):
    '''
    Reads in the log and then produces a dfg from it. The function then generates a Pandas DataFrame.

    The returned DataFrame is a pivot table, rows are source nodes, columns are target nodes and the values is the frequency the vertex has been used 
    '''

    df = get_df_from_dfg(log, activity_key=activity_key)

    # find events which are not in ingoing and outgoing events
    from_events = df["From"].unique()
    to_events = df["To"].unique()
    only_from_elements = list(set(from_events) - set(to_events))
    only_to_elements = list(set(to_events) - set(from_events))

    # add them to the dataframe in the column in which they were not in
    # that way all elements in the pivot table are in both column and rows
    for element in only_from_elements:
        df.loc[len(df)] = [from_events[0], element, 0]
    for element in only_to_elements:
        df.loc[len(df)] = [element, to_events[0], 0]

    df = df.pivot(index="From", columns="To", values="Frequency")
    df = df.fillna(0)

    return df

def get_weighted_df(df):
    '''
    Transforms the frequency of the used vertices to weights. The weight is the portion of all outgoing vertices of a node.
    Ex.: Node A has three vertices with frequency 8, 7 and 5. After applying the transformation the weights are 0.4, 0.35 and 0.25

    Returns the normalized df
    '''
    
    normalized_df = df.apply(lambda x: (x / x.sum()), axis="columns")
    normalized_df = normalized_df.fillna(0)

    return normalized_df