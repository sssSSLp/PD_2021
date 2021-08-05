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

## License
BSD 3-Clause License (see `LICENSE` file)