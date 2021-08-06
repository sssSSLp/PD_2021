# PD_2021


## How to train a model
To setup the training script, use pip.
```shell
pip3 install -e .
```

Then, you can train a classification model to classify PD.
```shell
cd script
python3 train_classifier.py --feature_dir ../data/ --feature_id 1 --background ../data/background_all.csv --n_trials 1 --out result --test_run --params best_classifier_params.yaml
```

Thre result to evaluate the trained model is here:
```shell
./result/result.out
```


## R codes
All R codes in this repository cover the preprocessing, normalization, analysis, and plotting of differential expression analysis in TITLE OF ARTICLE (URL). These R codes are dependent on the following packages:
- DESeq2
- reshape2
- dplyr
- ggplot2
- gplots
- ggbeeswarm


## Data
The csv files in this repository contain the original data used in the TITLE OF ARTICLE (URL). The description of each file is as follows:
- *`file name`: description*
- `background_all.csv`: Characteristics of all participants.
- `background_first.csv`: Characteristics of participants in the first cohort.
- `background_second.csv`: Characteristics of participants in the second cohort.
- `normalized_count_with_deseq2.csv`: Normalized read-count of all participants.
- `readcount_all.csv`: Original read-count data of all participants.
- `readcount_first.csv`: Original read-count data of the participants in first cohort.
- `readcount_secon.csv`: Original read-count data of the participants in second cohort.


## License
BSD 3-Clause License (see `LICENSE` file)
