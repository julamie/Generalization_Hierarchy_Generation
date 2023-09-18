echo "Retrieving information from Coselog event log..."
python get_traces_coselog.py    # returns coselog.csv
echo "--------------------------------------------------"

echo "Use MAFFT Vectorization to align traces"
mafft out/coselog.csv > out/coselog_mafft.csv   # returns coselog_mafft.csv
echo "--------------------------------------------------"

echo "Convert result to arx readable csv"
python coselog_Clustering.py    # returns coselog_for_arx_mafft.csv
echo "--------------------------------------------------"

echo "Naively vectorize traces"
python coselog_naive_encoding.py    # returns coselog_for_arx_naive.csv
echo "--------------------------------------------------"

echo "Anonymize using ARX"
cd ../../Component_2/
javac -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes.java
java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes "../Component_1/coselog/out/coselog_for_arx_mafft.csv" "../Component_1/coselog/coselog_hierarchy.csv" "../Component_1/coselog/out/coselog_anon_k_5.csv" 5 # returns coselog_anon_k_{k}.csv
echo "--------------------------------------------------"

cd ../Component_1/coselog/
echo "Create the anonymized event log"
python rewrite_traces_in_coselog.py     # returns coselog_anonymized_log_k_{k}.xes
echo "--------------------------------------------------"

echo "Anonymize the resource attribute"
cd ../../Component_2/
for file in ../Component_1/coselog/out/coselog_for_arx_resource_attribute_*.csv;
do
    echo "Anonymizing $file" 
    java -cp .:arx-3.9.1-gtk-64.jar ARXAnonymizeAttributes $file "../Component_1/coselog/resource_hierarchy.csv" "../Component_1/coselog/out/coselog_anon_resource_k_5.csv" 5 # returns coselog_anon_k_{k}.csv
done
echo "--------------------------------------------------"

echo "Combine anonymized logs to one big anonymized event log"
cd ../Component_3/
python rewrite_attributes.py