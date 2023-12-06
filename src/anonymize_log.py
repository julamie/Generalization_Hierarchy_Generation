import pm4py
import csv
import pandas as pd
from pm4py.statistics.traces.generic.log import case_statistics
import subprocess
import glob

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
        
        # assign a one letter code to every activity
        # example: "Process Form: A, Approve Form: B, etc."
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz!#$%&'()*+,-./:;?@[\]^_`{|}~"
        self.dict_convert = dict(zip(self.activities, letters[:len(self.activities)]))

        variants = []
        abstracted_variants = []
        for i in self.variants_length:
            variants.append(list(i['variant']))

        # convert the activities to letters using dict_convert
        for j in variants:
            local_list = []
            for l in j:
                local_list.append(self.dict_convert.get(l))
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

    def run_arx(self, k, hierarchy_file):
        """
        Runs the ARX Anonymizing tool. Function will generate a csv file with anonymized traces which
        are k-anonymized. k has to be known beforehand.

        Output file: [prefix]_anon_k_[k].csv
        """

        # compile java file
        subprocess.run("javac -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes.java", shell=True)

        # files needed are parameters for java program
        arx_file = f'out/{self.output_file_prefix}_for_arx_mafft.csv'
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
        self.df_priv_log = pd.concat(frames)
        event_log = pm4py.convert_to_event_log(self.df_priv_log)
        pm4py.write_xes(event_log, f"out/{self.output_file_prefix}_anonymized_log_k_{k}.xes")
        
        if attribute != "":
            # set of all unique attribute lengths
            unique_attribute_length = set(attribute_length)
            for size in unique_attribute_length:
                self.clean_attribute_name = ''.join(e for e in attribute if e.isalnum())
                with open(f"out/{self.output_file_prefix}_for_arx_attribute_{self.clean_attribute_name}_len_{size}.csv", "w") as f:
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

    def anonymize_attributes(self, attribute, hierarchy_path, k):
        """
        Finds all attribute traces of different lengths, created by rewrite_traces() and runs these through ARX again.
        If the number of traces of a certain length are smaller than k, all attribute values are changed to '*'.

        Output files: [prefix]_[attribute name]_len_[trace length]_anon_k_[k].csv
        """

        hierarchy_path = f"out/{hierarchy_path}"
        attribute_files = glob.glob(f"out/{self.output_file_prefix}_for_arx_attribute_{self.clean_attribute_name}_len_*.csv")
        
        for i, file in enumerate(attribute_files):
            file_df = pd.read_csv(file)
            # columns are named: variant, row_1, row_2, ..., row_n
            trace_length = file_df.shape[1] - 1
            number_of_rows = file_df.shape[0]

            print(f"Anonymizing attribute traces of length {trace_length}...")
            output_file = f'out/{self.output_file_prefix}_{self.clean_attribute_name}_len_{trace_length}_anon_k_{k}.csv'

            # if the number of rows if smaller than k, anonymize all values to '*'
            if number_of_rows < k:
                print(f"Number of rows (={number_of_rows}) is smaller than k (={k}). Anonymize all values to '*'")
                for column in file_df.columns:
                    if column == "variant":
                        continue
                    file_df[column] = '*'
                file_df.to_csv(output_file, index=False)
            else:
                command = f"java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes {file} {hierarchy_path} {output_file} {k}"
                subprocess.run(command, shell=True)
            print(f"Completed {i + 1}/{len(attribute_files)}")
        print("Complete ✓")
    
    def rewrite_trace_attributes_in_log(self, k, attribute):
        """
        Changes all attribute values in the previously anonymized log with the anonymized attribute values.

        Output file: [prefix]_anonymized_log_k_[k]_with_attributes.xes
        """

        self.clean_attribute_name = ''.join(e for e in attribute if e.isalnum())
        attribute_files = glob.glob(f"out/{self.output_file_prefix}_{self.clean_attribute_name}_len_*_anon_k_{k}.csv")

        logs = []
        for file_nr, file in enumerate(attribute_files):
            frames = []
            priv_attr_df = pd.read_csv(file)
            for i in priv_attr_df.values:
                trace = self.df_priv_log[self.df_priv_log['case:concept:name'] == i[0]]
                trace_a = []
                for j in range(1, len(i)):
                    trace_a.append(i[j])
                trace.drop(attribute, axis=1, inplace=True)
                trace[attribute] = trace_a
                frames.append(trace)
            sublog = pd.concat(frames)
            logs.append(sublog)
            print(f"Completed {file_nr + 1}/{len(attribute_files)}")
        
        total_logs = pd.concat(logs)
        event_log = pm4py.convert_to_event_log(total_logs)
        pm4py.write_xes(event_log, f"out/{self.output_file_prefix}_anonymized_log_k_{k}_with_attributes.xes")
        print("Complete ✓")

# --------------------------------------------------------

def create_activity_hierarchy_file(event_log_path, file_prefix, metric_name, activities_column="", length=0):
    """
    Uses a given metric and generates with it a generalization hierarchy used for anonymizing a given event log.
    If Role Comparison is the metric to be used, activities_column needs to be defined
    If N_gram is the metric to be used, the maximum length to compare needs to be definined
    
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
    elif metric_name == "Simple_Role_Similarity":
        metric = Role_Comparison.Role_Comparison(log, activities_column=activities_column)
        metric.perform_clustering(weighted=False)
    elif metric_name == "Weighted_Role_Similarity":
        metric = Role_Comparison.Role_Comparison(log, activities_column=activities_column)
        metric.perform_clustering(weighted=True)
    elif metric_name == "Label_Similarity":
        metric = Label_Similarity.Label_Similarity(log)
        metric.perform_clustering()
    else:
        print(f"Incorrect metric name {metric_name}! Allowed names are: "
              "Simple_Jaccard, Weighted_Jaccard, " 
              "Simple_Jaccard_N_Gram, Weighted_Jaccard_N_Gram, "
              "Simple_Role_Similarity, Weighted_Role_Similarity and "
              "Label_Similarity")
        return None
    
    hierarchy_file_path = f"{file_prefix}_{metric_name}_hierarchy.csv"
    Clustering.generate_hierarchy_file_with_dummies(metric.activities, metric.distance_matrix, metric.linkage, hierarchy_file_path)
    
    return hierarchy_file_path

def create_attribute_hierarchy_file(event_log_path, file_prefix, attribute_key, metric_name, length=0):
    log = Log_processing.get_log(event_log_path)

    if metric_name == "Simple_Jaccard":
        metric = Jaccard.Simple_Jaccard(log)
        metric.perform_clustering(activity_key=attribute_key)
    elif metric_name == "Weighted_Jaccard":
        metric = Jaccard.Weighted_Jaccard(log)
        metric.perform_clustering(activity_key=attribute_key)
    elif metric_name == "Simple_Jaccard_N_Gram":
        metric = Jaccard.Jaccard_N_grams(log)
        metric.perform_clustering(activity_key=attribute_key, length=length)
    elif metric_name == "Weighted_Jaccard_N_Gram":
        metric = Jaccard.Weighted_Jaccard_N_grams(log)
        metric.perform_clustering(activity_key=attribute_key, length=length)
    elif metric_name == "Label_Similarity":
        metric = Label_Similarity.Label_Similarity(log)
        metric.perform_clustering(activity_key=attribute_key)
    else:
        print(f"Incorrect metric name {metric_name}! Allowed names are: "
              "Simple_Jaccard, Weighted_Jaccard, " 
              "Simple_Jaccard_N_Gram, Weighted_Jaccard_N_Gram and "
              "Label_Similarity")
        return None
    
    clean_attribute_name = ''.join(e for e in attribute_key if e.isalnum())
    hierarchy_file_path = f"{file_prefix}_attribute_{clean_attribute_name}_{metric_name}_hierarchy.csv"
    Clustering.generate_hierarchy_file_with_dummies(metric.activities, metric.distance_matrix, metric.linkage, hierarchy_file_path)
    
    return hierarchy_file_path

def anonymize_log(event_log_path, output_file_prefix, k, act_hierarchy_path, attribute="", attr_hierarchy_path=""):
    """
    Runs all the necessary steps used for anonymizing a given event_log.
    To anonymize an attribute too, you need to define the attribute parameter.

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
    log.run_arx(k, act_hierarchy_path)
    print("--------------------------------------------------")

    print("Create the anonymized event log. This could take a while...")
    log.rewrite_traces_in_log(k, attribute=attribute)
    print("--------------------------------------------------")

    if attribute != "":
        print("Anonymize the given attribute")
        log.anonymize_attributes(attribute, attr_hierarchy_path, k)
        print("--------------------------------------------------")

        print("Create the anonymized event log with anonymized attributes. This could take a while...")
        log.rewrite_trace_attributes_in_log(k, attribute)
        print("--------------------------------------------------")

        print(f"Anonymization complete. File can be found at out/{output_file_prefix}_anonymized_log_k_{k}_with_attributes.xes")
    else:
        print(f"Anonymization complete. File can be found at out/{output_file_prefix}_anonymized_log_k_{k}.xes")

# --------------------------------------------------------

if __name__ == "__main__":
    log_path = "../logs/sepsis_event_log.xes"
    file_prefix = "Sepsis"
    k = 15
    metric_name = "Simple_Jaccard"
    act_hierarchy_path = create_activity_hierarchy_file(log_path, file_prefix, metric_name)
    attr_hierarchy_path = create_attribute_hierarchy_file(log_path, file_prefix, "org:group", metric_name)
    anonymize_log(log_path, file_prefix, k, act_hierarchy_path, attribute="org:group", attr_hierarchy_path=attr_hierarchy_path)