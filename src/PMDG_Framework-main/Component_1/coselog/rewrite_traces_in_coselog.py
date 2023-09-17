import itertools
import time

import numpy as np
import pandas
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import distance
import csv
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import pm4py
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.filtering import *

pd.set_option('mode.chained_assignment', None)


class RewriteLog:
    def __init__(self, log, privacy_log, attribute):
        self.log = log
        self.privacy_log = privacy_log
        self.attribute = attribute

    def writer(self):
        df_log = pm4py.convert_to_dataframe(self.log)
        df_log['concept:name'] = df_log['concept:name'] + " - " + df_log['lifecycle:transition']
        df_log = pm4py.filter_variants_by_coverage_percentage(df_log, 0.00014)
        frames = []
        attribute_writer = []
        attribute_length = []
        for i in self.privacy_log.values:
            # trace is one row of values in privacy log
            trace = df_log[df_log['case:concept:name'] == i[0]]
            trace_a = []
            attribute_variant = []
            # append all events which are not "-" to trace_a
            for j in range(1, len(i)):
                if i[j] == '-':
                    continue
                else:
                    trace_a.append(i[j])
            # if values were "-" replace these with trace_a
            if len(trace['concept:name']) != len(trace_a):
                pass
                #print(trace['concept:name'])
                #print(trace_a)
            else:
                trace.drop('concept:name', axis=1, inplace=True)
                trace['concept:name'] = trace_a
            
            # add trace to frames and variant name to attribute variant
            frames.append(trace)
            attribute_variant.append(i[0])
            
            # add all values of the given attribute to attribute_variant
            for k in trace[self.attribute].values.tolist():
                attribute_variant.append(k)
            # add the list of the variant plus all values of the attribute to attribute_writer 
            attribute_writer.append(attribute_variant)
            attribute_length.append(len(attribute_variant))
            
        # convert all traces in the list frames to an event log
        new_df = pd.concat(frames)
        event_log = pm4py.convert_to_event_log(new_df)
        pm4py.write_xes(event_log, "coselog_anonymized_log_k_5.xes")
        
        # set of all unique attribute lengths
        '''
        unique_attribute_length = set(attribute_length)
        for size in unique_attribute_length:
            with open("output_for_arx_2_" + str(size) + ".csv", "w") as f:
                for trace in attribute_writer:
                    if len(trace) == size:
                        f.write("%s\n" % trace)
        '''
        
        # return event_log

log = xes_importer.apply(str(pathlib.Path().resolve()) + "/coselog.xes")
privacy_log = pd.read_csv(str(pathlib.Path().resolve()) + "/coselog_anon_k_5.csv")

writer = RewriteLog(log, privacy_log, 'org:resource')
writer.writer()
