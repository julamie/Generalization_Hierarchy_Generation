import pm4py

def get_event_log(path: str):
    return pm4py.read_xes(path)

def save_dfg_of_event_log(event_log):
    dfg, start_activities, end_activities = pm4py.discovery.discover_dfg(event_log, 
                                                                         case_id_key= "case:concept:name",
                                                                         activity_key= "concept:name")
    pm4py.vis.save_vis_dfg(dfg, start_activities, end_activities, file_path="../out/dfg/sepsis.svg")

if __name__ == "__main__":
    event_log = get_event_log("../logs/sepsis_event_log.xes")
    save_dfg_of_event_log(event_log)
    