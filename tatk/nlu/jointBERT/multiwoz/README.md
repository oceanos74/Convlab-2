# BERTNLU on multiwoz

Based on pre-trained bert, BERTNLU use an MLP for slot tagging and another MLP for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `{"Hotel-Inform":[["Price", "cheap"]]}`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. An MLP takes bert word embeddings as input and classify the tag label. If you set `context=true` in config file, utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for classification.  
- For each of the other dialog acts, such as `(Hotel-Request, Address, ?)`, another MLP takes embeddings of `[CLS]` of current utterance as input and do the binary classification. If you set `context=true` in config file, utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for classification.  

We fine-tune BERT parameters on multiwoz.

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `bert/multiwoz` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data/` dir.

#### Train a model

On `bert` dir:

```sh
$ python train.py --config_path multiwoz/configs/[config_file]
```

The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. 

#### Predict

See `nlu.py` for usage

#### Trained model

We have trained two models: one use context information (`configs/multiwoz_all_context.json`) and the other doesn't (`configs/multiwoz_all.json`) on **all** utterances of multiwoz dataset (`data/multiwoz/[train|val|test].json.zip`). Performance:

|                 | All Data (Slot/Intent/Overall F1) | User Data (Slot/Intent/Overall F1) |
| --------------- | --------------------------------- | ---------------------------------- |
| without context | 62.83 / 83.01 / 70.37             | 82.15 / 82.23 / 82.18              |
| with context    | 63.72 / 87.35 / 72.45             | 83.74 / 89.33 / 85.70              |

Models can be download form:

Without context: https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip

With context: https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all_context.zip



## Data

We use the multiwoz data (`data/multiwoz/[train|val|test].json.zip`).

## References

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}
```
