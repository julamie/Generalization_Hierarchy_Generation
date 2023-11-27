import pm4py
import string
import csv
import pandas as pd
from pm4py.statistics.traces.generic.log import case_statistics
import subprocess

import Jaccard, Role_Comparison, Label_Similarity
import Log_processing, Clustering

"""
This file combines all steps necessary for anonymizing an event log using the PMDG Framework, into one single function.
It also generates the automatic generalization hierarchies.

The original files were written by Ryan Hildebrant. 
You can find the original project at: https://github.com/Ryanhilde/PMDG_Framework/

"""

class Event_log:
    def __init__(self, event_log_path, output_file_prefix):
        self.event_log_path = event_log_path
        self.output_file_prefix = output_file_prefix

        self.df_log = Log_processing.get_log(event_log_path)

        # trim log if necessary
        self.df_log = pm4py.filter_variants_by_coverage_percentage(self.df_log, 0.001)
        variants_count = case_statistics.get_variant_statistics(self.df_log)
        self.variants_length = sorted(variants_count, key=lambda x: x['count'], reverse=True)

        # get list of all activities in log
        self.activities = list(pm4py.get_event_attribute_values(self.df_log, "concept:name").keys())

    def convert_to_mafft_output_file(self):
        """
        Generates a csv file where the names of the activities are changed to a letter of the alphabet.
        This file will be used by MAFFT for the Trace Vectorization step.

        Output file: [prefix].csv
        """
        
        # designate a letter of the alphabet to every activity
        # example: "Process Form: A, Approve Form: B, etc."
        self.dict_convert = dict(zip(self.activities, string.ascii_uppercase[:len(self.activities)]))

        variants = []
        abstracted_variants = []
        for i in self.variants_length:
            variants.append(list(i['variant']))

        # convert the activities to letters using dict_convert
        for j in variants:
            local_list = []
            for k in j:
                local_list.append(self.dict_convert.get(k))
            abstracted_variants.append(local_list)

        # write to file for mafft vectorization
        counter = 0
        with open(f"out/{self.output_file_prefix}.csv", "w+") as f:
            for row in abstracted_variants:
                f.write("> " + str(counter) + '\n')
                f.write("%s\n" % ''.join(str(col) for col in row))
                counter += 1
        print("Complete ✓")

    def run_mafft(self):
        """
        Calls MAFFT to perform trace vectorization.

        Output file: [prefix]_mafft.csv
        """
        
        subprocess.run(f"mafft --text out/{self.output_file_prefix}.csv > out/{self.output_file_prefix}_mafft.csv", shell=True)
        print("Complete ✓")

    def convert_mafft_file_to_arx_output_file(self):
        """
        Prepares event log for anonymization using ARX. This function converts the data from the MAFFT file to a csv file
        readible by the ARX Anonymizing tool.

        Output file: [prefix]_for_arx_mafft.csv
        """
        
        writer = []

        dict_convert_reverse = {y: x for x, y in self.dict_convert.items()}
        dict_convert_reverse['-'] = '-'

        with open(f'out/{self.output_file_prefix}_mafft.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        variant_pairs = {}

        filter_variants = self.df_log.groupby("case:concept:name")['concept:name'].apply(lambda tags: ','.join(tags))

        for i in filter_variants.unique():
            variant_pairs[i] = filter_variants.index[filter_variants == i].tolist()

        for i in data[1::2]:
            current_variant = []
            for j in i:
                variant = ''.join(map(str, j))
                for i in variant:
                    current_variant.append(dict_convert_reverse.get(i))
            cleaned_variant = [i for i in current_variant if i != '-']
            trace_id = variant_pairs.get(','.join(map(str, cleaned_variant)))

            for i in trace_id:
                trace_instance = []
                trace_instance.append(i)
                trace_instance.extend(current_variant)
                writer.append(trace_instance)

        # determine number of columns in csv file
        # header will be: variant,row_1,row_2,row_3,...
        number_rows = len(writer[0]) # get the length of an arbitrary trace
        header = "variant"
        for i in range(1, number_rows):
            header += f",row_{i}"

        with open(f"out/{self.output_file_prefix}_for_arx_mafft.csv", "w") as f:
            f.write(header + "\n")
            for trace in writer:
                f.write("%s\n" % ', '.join(trace))
        print("Complete ✓")

    def run_arx(self, k, hierarchy_file=None):
        """
        Runs the ARX Anonymizing tool. Function will generate a csv file with anonymized traces which
        are k-anonymized. k has to be known beforehand.

        Output file: [prefix]_anon_k_[k].csv
        """

        # compile java file
        subprocess.run("javac -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes.java", shell=True)

        # files needed are parameters for java program
        arx_file = f'out/{self.output_file_prefix}_for_arx_mafft.csv'
        if not hierarchy_file:
            hierarchy_file = f'out/{self.output_file_prefix}_hierarchy.csv'
        else:
            hierarchy_file = f'out/{hierarchy_file}'
        output_file = f'out/{self.output_file_prefix}_anon_k_{k}.csv'

        # run ARX Java file
        command = f"java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes {arx_file} {hierarchy_file} {output_file} {k}"
        subprocess.run(command, shell=True)
        print("Complete ✓")

    def rewrite_traces_in_log(self, k, attribute=""):
        """
        Combines all the k-anonymized traces to an XES file, e.g. it generates an event log file.
        If necessary, this function also generates more files for ARX to anonymize, if you want to
        anonymize the attributes too.

        Output file: [prefix]_anonymized_log_k_[k].xes
        """

        pd.set_option('mode.chained_assignment', None)

        privacy_log = pd.read_csv(f'out/{self.output_file_prefix}_anon_k_{k}.csv')
        frames = []
        attribute_writer = []
        attribute_length = []
        counter = 0

        for i in privacy_log.values:
            # print progress
            counter += 1
            if counter % 100 == 0:
                print(f"{counter}/{len(privacy_log.values)} traces processed")
            
            # trace is one row of values in privacy log
            trace = self.df_log[self.df_log['case:concept:name'] == i[0]]
            trace_a = []
            attribute_variant = []
            # append all events which are not "-" to trace_a
            for j in range(1, len(i)):
                if i[j] != '-':
                    trace_a.append(i[j])
            # if values were "-", replace these with trace_a
            if len(trace['concept:name']) == len(trace_a):
                trace.drop('concept:name', axis=1, inplace=True)
                trace['concept:name'] = trace_a
            
            # add trace to frames and variant name to attribute variant
            frames.append(trace)
            attribute_variant.append(i[0])
            
            # add all values of the given attribute to attribute_variant
            if attribute != "":
                for value in trace[attribute].values.tolist():
                    attribute_variant.append(value)
                # add the list of the variant plus all values of the attribute to attribute_writer 
                attribute_writer.append(attribute_variant)
                attribute_length.append(len(attribute_variant))
            
        # convert all traces in the list frames to an event log
        new_df = pd.concat(frames)
        event_log = pm4py.convert_to_event_log(new_df)
        pm4py.write_xes(event_log, f"out/{self.output_file_prefix}_anonymized_log_k_{k}.xes")
        
        if attribute != "":
            # set of all unique attribute lengths
            unique_attribute_length = set(attribute_length)
            for size in unique_attribute_length:
                with open(f"out/{self.output_file_prefix}_for_arx_resource_attribute_" + str(size) + ".csv", "w") as f:
                    # write top line for csv file
                    f.write("variant")
                    for i in range(1, size):
                        f.write(",row_" + str(i))
                    f.write("\n")

                    # write traces of length size to file
                    for trace in attribute_writer:
                        if len(trace) == size:
                            f.write("%s\n" % ', '.join(trace))
        print("Complete ✓")

# --------------------------------------------------------

def create_hierarchy_file(event_log_path, file_prefix, metric_name, activities_column="", length=0):
    """
    Uses a given metric and generates with it a generalization hierarchy used for anonymizing a given event log.

    Output file: [prefix]_[metric]_hierarchy.csv
    """
    
    log = Log_processing.get_log(event_log_path)

    if metric_name == "Simple_Jaccard":
        metric = Jaccard.Simple_Jaccard(log)
        metric.perform_clustering()
    elif metric_name == "Weighted_Jaccard":
        metric = Jaccard.Weighted_Jaccard(log)
        metric.perform_clustering()
    elif metric_name == "Simple_Jaccard_N_Gram":
        metric = Jaccard.Jaccard_N_grams(log)
        metric.perform_clustering(length=length)
    elif metric_name == "Weighted_Jaccard_N_Gram":
        metric = Jaccard.Weighted_Jaccard_N_grams(log)
        metric.perform_clustering(length=length)
    elif metric_name == "Weighted_Role_Similarity":
        metric = Role_Comparison.Role_Comparison(log, activities_column=activities_column)
        metric.perform_clustering(weighted=True)
    elif metric_name == "Simple_Role_Similarity":
        metric = Role_Comparison.Role_Comparison(log, activities_column=activities_column)
        metric.perform_clustering(weighted=False)
    elif metric_name == "Label_Similarity":
        metric = Label_Similarity.Label_Similarity(log)
        metric.perform_clustering()
    
    Clustering.generate_hierarchy_file_with_dummies(metric.activities, metric.distance_matrix, metric.linkage, f"{file_prefix}_{metric_name}_hierarchy.csv")

def anonymize_log(event_log_path, output_file_prefix, k, hierarchy_file_path=None):
    """
    Runs all the necessary steps used for anonymizing a given event_log.

    Output file: [prefix]_anonymized_log_k_[k].xes
    """

    log = Event_log(event_log_path, output_file_prefix)

    print("Retrieving information from event log...")
    log.convert_to_mafft_output_file()
    print("--------------------------------------------------")

    print("Use MAFFT Vectorization to align traces")
    log.run_mafft()
    print("--------------------------------------------------")

    print("Convert result to arx readable csv")
    log.convert_mafft_file_to_arx_output_file()
    print("--------------------------------------------------")

    print("Anonymize using ARX. This could take a little bit longer...")
    log.run_arx(k, hierarchy_file=hierarchy_file_path)
    print("--------------------------------------------------")

    print("Create the anonymized event log. This could take a while...")
    log.rewrite_traces_in_log(k)
    print("--------------------------------------------------")

    print(f"Anonymization complete. File can be found at out/{output_file_prefix}_anonymized_log_k_{k}.xes")

# --------------------------------------------------------

if __name__ == "__main__":
    log_path = "../logs/sepsis_event_log.xes"
    file_prefix = "Sepsis"
    metric_name = "Simple_Jaccard"
    create_hierarchy_file(log_path, file_prefix, metric_name)
    anonymize_log(log_path, file_prefix, 5, hierarchy_file_path=f"{file_prefix}_{metric_name}_hierarchy.csv")