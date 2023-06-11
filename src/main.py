import pm4py

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

if __name__ == "__main__":
    log = get_log("../logs/sepsis_event_log.xes")
    print_log_info(log)
    #save_dfg_of_log(log)
    