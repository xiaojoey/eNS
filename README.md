# Detect Noonan Syndrome via Deep Learning
This project utilizes diagnosis texts extracted from electronic health records to classify Noonan Syndrome.

This repo contains the source code used to generate the data for the manuscript. 

## Abstract

### Purpose
 The variable expressivity and multisystem features of Noonan syndrome (NS) make it difficult for patients to obtain a timely diagnosis. Genetic testing can confirm a diagnosis, but underdiagnosis is prevalent due to a lack of recognition and referral for testing. Our study investigated the utility of using electronic health records (EHR) to identify patients at high risk of NS.
### Methods
 Using diagnosis texts extracted from Cincinnati Children's Hospital’s EHR database, we constructed deep learning models from 162 NS cases and 16,200 putative controls. Performance was evaluated on two independent test sets, one containing NS patients who were previously diagnosed and the other containing patients with undiagnosed NS. 
### Results
Our novel method performed significantly better than the previous method, with the convolutional neural network model achieving the highest area under the precision-recall curve in both test sets (diagnosed: 0.43, undiagnosed: 0.17).
### Conclusion
The results suggested the validity of utilizing text-based deep learning methods to analyze EHR and demonstrated the value of this approach as a potential tool to identify patients with features of rare diseases. Given the paucity of medical geneticists, this has the potential to reduce disease underdiagnosis by prioritizing patients who will benefit most from a genetics referral.

