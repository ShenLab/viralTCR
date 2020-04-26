# viralTCR
A machine learning method to predict TCR-peptide binding and the method's application in COVID-19

## Abstract
### Background
1. T cells play a central role in viral response and vaccination.
2. T cell receptor(TCR) - peptide binding provides specificity.
### Significance
1. By predicting TCR-peptide binding, one can "reverse engineer" the source peptides that triggered T cell response in patients.
2. Vaccine design: strong response from T cells is required for maturation and proliferation of antibody-producing B cells.
### Project Aim
1. Predict binding specificity of TCR and peptides by machine learning.
2. Search for SARS-CoV-2 peptides that elicit strong T cell response based on TCRs from COVID-19 patients. 
### Results
1. Classification Model 10-folds Acc:77.20% ± 0.78%.
2. 2219 SARS-CoV-2 peptides have high binding prob (score >0.9) among COVID-19 patients.
3. Excluding healthy control,16 peptides appeared in more than 6 samples.
### Conclusions
1. We proposed a supervised deep learning method to predict TCR-peptide binding.
2. COVID-19 patients share more peptides with each other than with healthy individual, based on prediction given TCR sequences.
3. Shared peptides across patients could be candidates for rational vaccine design.

## Data
### Training and testing
TCR specificity ([VDJdb](https://vdjdb.cdr3.net/search)), also, see downloaded VDJdb_TCR.tsv
### Application
[10 patients TCR repertoire](https://www.medrxiv.org/content/10.1101/2020.03.15.20033472v1.supplementary-material), also, see TCR.csv
[SARA-CoV-2 sequence](https://www.ncbi.nlm.nih.gov/nuccore/MN908947), also, see virus_sequence.txt
### Verify
Sequence homology and MHC binding ability predicted [epitopes](https://www.cell.com/cell-host-microbe/pdf/S1931-3128(20)30166-9.pdf?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1931312820301669%3Fshowall%3Dtrue)

## Method
### [TCRCDR](http://opig.stats.ox.ac.uk/webapps/stcrpred/CDRPred#download_software)
Predictor for five of the six complementarity-determining regions (B1, B2, A1, A2 and A3) on an T-cell receptor (TCR)
### [SeqVec](https://github.com/Rostlab/SeqVec)
Embedding six complementarity-determining regions and peptide
### LSTM Autoencoder
Transfering embedded TCR-peptide pairs to reasonable dimension
### DNN
Training model for TCR-peptide pairs binary classification

Average ROC for repeated 10-fold cross validation:
![alt text](https://github.com/ShenLab/viralTCR/blob/master/auroc.jpeg)
