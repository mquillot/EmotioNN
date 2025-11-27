# EmotioNN
Textual Emotion Recognition Neural Network


# Download the dataset
You will find the dataset I used here:
[Emotion Dataset from HuggingFace](https://huggingface.co/datasets/dair-ai/emotion)


At the end, you should have three files in `data/Emotion-detection-from-text` subfolder: test.csv, training.csv and validation.csv.

Each CSV file must contain two columns, `text` and `value`. 

# UV Configuration for MAC OS ARM
For this project, I am using my personal MacBook Pro M1. The M1 uses an ARM processor. X86 programs can be launched transparently in your ARM computer so you must ensure that you installed the right python version to have the best performance your processor can offer.

For dependency management, I use UV. With UV, you can specify the right version of python you want to use. So, if like me you are using an ARM processor (M1, M2, ...) and UV, please use the following uv commands:

```
uv python pin cpython-3.14.0-macos-aarch64-none
```

Else, you will use the x86 by default.


# TODO list

- [ ] Add isort as automatic import sorting
- [ ] Implement Early Stopping
- [ ] Track training measures with a solution like Tensorboard
- [x] Add black as python formatter