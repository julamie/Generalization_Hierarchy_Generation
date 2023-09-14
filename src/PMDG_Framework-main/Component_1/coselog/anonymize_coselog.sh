echo "Retrieving information from Coselog event log..."
python get_traces_coselog.py    # returns coselog.csv
echo "Retrieval completed"

echo "Use MAFFT Vectorization to align traces"
mafft coselog.csv > coselog_mafft.csv   # returns coselog_mafft.csv
echo "Traces aligned"

echo "Convert result to arx readable csv"
python coselog_Clustering.py    # returns coselog_for_arx_mafft.csv
echo "Conversion complete"

echo "Naively vectorize traces"
python coselog_naive_encoding.py    # returns coselog_for_arx_naive.csv
echo "Naively vectorized"

echo "Anonymize using ARX"
cd ../../Component_2/
javac -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes.java
java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes "../Component_1/coselog/coselog_for_arx_mafft.csv" "../Component_1/coselog/coselog_hierarchy.csv" "../Component_1/coselog/coselog_anon_k_5.csv" 5  # returns coselog_anon_k_{k}.csv

cd ../Component_1
echo "Create the anonymized event log"
python rewrite_traces_in_log.py     # returns coselog_anonymized_log_k_{k}.xes
