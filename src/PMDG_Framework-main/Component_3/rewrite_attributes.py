import itertools
import os
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


class RewriteAttributes:
    def __init__(self, log, privacy_log, attribute):
        self.log = log
        self.privacy_log = privacy_log
        self.attribute = attribute

    def writer(self):
        df_log = pm4py.convert_to_dataframe(self.log)
        df_log['concept:name'] = df_log['concept:name'] + " - " + df_log['lifecycle:transition']
        frames = []
        for i in self.privacy_log.values:
            trace = df_log[df_log['case:concept:name'] == i[0]]
            trace_a = []
            for j in range(1, len(i)):
                trace_a.append(i[j])
            trace.drop(self.attribute, axis=1, inplace=True)
            trace[self.attribute] = trace_a
            frames.append(trace)
        new_df = pd.concat(frames)
        return new_df

# load event log
log = xes_importer.apply(str(pathlib.Path().resolve()) + "/../Component_1/coselog/coselog.xes")
logs = []

k = 5
alignment = "Vectorization"
attribute = "org:resource"
directory = "../Component_1/coselog/out"

for file in os.listdir(directory):
    privacy_log = pd.read_csv(directory + "/" + os.fsdecode(file))
    print(directory + "/" + os.fsdecode(file))
    writer = RewriteAttributes(log, privacy_log, attribute)
    sublog = writer.writer()

    logs.append(sublog)

# combine anonymized logs to one big event log
total_logs = pd.concat(logs)
event_log = pm4py.convert_to_event_log(total_logs)
pm4py.write_xes(event_log, 'Alignments/BPIC 2013 ' + alignment + ' ' + attribute + '/bpic_' + alignment +
                '_fully_aligned_k_' + str(k) + '.xes')
