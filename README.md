# In-silico-dengue-virus-antibody-prediction-2022
This Github repository is for our paper "Machine-learning-assisted high-throughput identification of potent and stable neutralizing antibodies against all four dengue virus serotypes" 
published in Scientific Reports. To read the paper please visit the following link: https://www.nature.com/articles/s41598-024-67487-8

# Authors
The work was done by Piyatida Natsrita, Phasit Charoenkwan, Watshara Shoombuatong, Panupong Mahalapbutr, Kiatichai Faksri, Sorujsiri Charoensudjai, Thanyada Rungrotmongkol and Chonlatip Pipattanaboon.

# Acknowledgement
This research project was financially supported by the Young Researcher Development Project of Khon Kaen University, Thailand.

# Dataset
We built a novel dataset (n = 1108) by compiling the interactions of CDR-H3 and epitope sequences with the half maximum inhibitory concentration (IC50) values, which represent neutralizing activities.

# Method
We used three different feature-encoding methods and ten ML algorithms to compare and exhibit the best performing model to determine the NAbs that can neutralize all 4 serotypes of DENV in unseen antibodies. These antibodies were further characterized for their binding sites, binding affinities, and binding stabilities using molecular docking and MD simulation.
More details about the methods can be found in the paper.

# Running the code
To run the code on your custom data just replace the filename at the appropriate commented place in the code. After installing the packages simply type the following in your command line.

```bash
python Ab_DENV_SequenceFeature.py
python Ab_DENV_AtomFeature.py
python Ab_DENV_FingerprintFeature.py
