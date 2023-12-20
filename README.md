# Generation of generalization hierarchies
In this repository is the code which was used to generate automatic generalization hierarchies for activities and attributes of event logs. 
Generalization hierarchies are needed to run the PMDG Framework, written by Ryan Hildebrant. 
You can find the PMDG Framework here: https://github.com/Ryanhilde/PMDG_Framework/tree/main

# Requirements
* MAFFT (https://mafft.cbrc.jp/alignment/software/)
* NetworkX (https://networkx.org/documentation/latest/install.html)  
* NumPy (https://numpy.org/install/)
* Pandas 1.5.X (https://pandas.pydata.org/pandas-docs/version/1.5.3/getting_started/install.html)  
Newer versions of Pandas sometimes throw errors when reading in newly created event logs. I recommend to keep using the older version
* pm4py (https://pm4py.fit.fraunhofer.de/install)
* scikit-learn (https://scikit-learn.org/stable/install.html)  
* sciPy (https://scipy.org/install/)  
* SpaCy (https://spacy.io/usage)  

# What's in this repository?
* **data**:  Certain functions take a bit of time to complete. That is why some results of the events logs were prerun and saved in this folder  
* **logs**:  The used event logs, are saved in here  
* **src**:   All of the source code is stored in here  
  - **src/Clustering.ipynb**: A Jupyter Notebook where all operations on the events logs were executed  
  - **src/Anonymization.py**: The shortened version of the PMDG Framework, that runs with the created generalization hierarchies  
  - **src/Log_processing.py**: All functions concerning reading in and working with an event log  
  - **src/Jaccard.py, Role_Comparison.py and Label_Similarity.py**: Implementations of the different distance metrics  
  - **src/Comparison.py**: All functions used for generating dendrograms, Fowlkes-Mallows-Scores und Handover-Graphs  
  - **src/ARXAnonymizeAttribute.java, arx-3.9.1-gtk-64.jar**: The ARX Anonymization Tool. This was part of the PMDG Framework and performs the anonymization of the event log  

# Who do I contact?
If you wish to contact me for any questions, you can use the following address: j.laabs01jl@gmail.com
