echo "Retrieving information from BPIC event log..."
python get_traces_bpic_2013.py    # returns bpic_2013.csv
echo "Retrieval completed"

echo "Use MAFFT Vectorization to align traces"
mafft bpic_2013.csv > bpic_2013_mafft.csv   # returns bpic_2013_mafft.csv
echo "Traces aligned"

echo "Naively vectorize traces"
python bpic_2013_naive_encoding.py    # returns bpic_2013_for_arx_naive.csv
echo "Naively vectorized"