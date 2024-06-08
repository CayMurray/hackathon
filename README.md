# Computational Neuroscience Hackathon
June 7, 2024 - University of Calgary

## Contributors
- Cayden Murray
- Shannon Pokojoy
- Shayan Shahrokhi
- Emily Kiddle

## Data
146 patients containing 72 schizophrenia patients, and 74 controls.

All data taken from https://figshare.com/articles/dataset/Cobre_Connectomes_GZ/1328237

Relevant articles: 
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9636375/
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2848257/#:~:text=In%20conclusion%2C%20we%20suggest%20that,important%20in%20normal%20brain%20function
- https://figshare.com/articles/dataset/MIST_A_multi-resolution_parcellation_of_functional_networks/5633638

## Analyses
- Principal Component Analysis with scatter plot, and Kolmogorov smirnov 2 sample test using euclidean distances between points of the same group, and euclidean distances between points of different groups.
- Average centrality differences to determine the importance of a node in influencing network activity such as shortest paths between vector points and eigen vector centrality.
- Simulation of differential equations to determine how each of the ROIs change over time, assuming function with periodic activity and using kuramoto order paramter values.

## Motivation
Schizophrenia is a “ behavioural disorder characterized by delusions, hallucinations, disorganized speech, blunted emotion, agitation or immobility and a host of associated symptoms (Kolb et al., 2019).” Being able to identify abnormalities in brain function can help with identification, prediction and treatment of schizophrenia.

## Hypothesis
We predicted significant differences between the population with schizophrenia and the healthy population, indicating that schizophrenic patients have differences in functional connectivity in comparison to controls.

## Results
See result figures for each section of data. There were significant differences found between the schizophrenia group and control group for all analyses performed, indicating that schizophrenic patients display abnormalities in functional connectivity through different regions of the brain.

## Running the tests yourself
To run the tests yourself, download `connectomes_cobre_scale_444.npy` as well as `subjects.txt` from the website link above containing all the data. Ensure that the data is in the directory ./workspace/data, creating the directory if necessary. All tests are found in the worspace directory.