# Physics5680FinalProject
Code for the Final Project of Physics 5680 AU24

Files:
- DataExploration.ipynb: Not relevant for methods/results; investigating structure of the patient, evidence and condition tables, some histograms, also contains first attempt at one-hot encoding that was very inefficient and got replaced by a different approach in "Features"
- Features.ipynb: one-hot encoding of the train patient file
- Random Forest.ipynb: building Random Forest Classifier, train on subset to investigate performance metrics, feature importance
- RandomForest.py
- CNN.ipynb: Building the architecture of the CNN and trying out different models based on a subset
- CNN.py: training of model on full dataset
- ImportCNN.ipynb: Importing the two models that were trained on the whole data and comparing their results and training process (time, epochs, accuracy; from .o files)
- EvidenceCondition.ipynb: Used as a lookup during feature importance analysis, feature names are ID numbers to put context to what each number means need to look into table that defines evidence
