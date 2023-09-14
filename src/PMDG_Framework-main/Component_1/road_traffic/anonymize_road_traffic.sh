echo "Retrieving information from Road Traffic Fine event log..."
python get_traces_road_traffic.py   # returns road_traffic.csv
echo "Retrieval completed"

echo "Use MAFFT Vectorization to align traces"
mafft road_traffic.csv > road_traffic_mafft.csv     # returns road_traffic_mafft.csv
echo "Traces aligned"

echo "Convert result to arx readable csv"
python road_traffic_Clustering.py   # returns road_traffic_for_arx_mafft.csv
echo "Conversion complete"

echo "Naively vectorize traces"
python road_traffic_naive_encoding.py   # returns road_traffic_for_arx_naive.csv
echo "Naively vectorized"

echo "Anonymize using ARX"
echo "Currently manual"