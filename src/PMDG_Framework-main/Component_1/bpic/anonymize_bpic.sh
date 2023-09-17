echo "Retrieving information from BPIC event log..."
python get_traces_bpic_2013.py    # returns bpic_2013.csv
echo "Retrieval completed"

echo "Use MAFFT Vectorization to align traces"
mafft bpic_2013.csv > bpic_2013_mafft.csv   # returns bpic_2013_mafft.csv
echo "Traces aligned"

echo "Convert result to arx readable csv"
python bpic_2013_Clustering.py    # returns bpic_for_arx_mafft.csv
echo "Conversion complete"

echo "Naively vectorize traces"
python bpic_2013_naive_encoding.py    # returns bpic_2013_for_arx_naive.csv
echo "Naively vectorized"

echo "Anonymize using ARX"
cd ../../Component_2/
javac -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes.java
java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes "../Component_1/bpic/bpic_2013_mafft.csv" "../Component_1/bpic/bpic_hierarchy.csv" "../Component_1/bpic/bpic_anon_k_5.csv" 5 # returns bpic_anon_k_{k}.csv

cd ../Component_1/bpic
echo "Create the anonymized event log"
python rewrite_traces_in_bpic.py     # returns bpic_anonymized_log_k_{k}.xes
