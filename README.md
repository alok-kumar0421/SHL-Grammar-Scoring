# SHL Grammar Scoring Engine

This project was developed as part of the SHL Research Intern assessment. The goal of the task is to predict a grammar score between 0 and 5 from spoken English audio.

I tried different audio-based features and models during the development process and finally used a combination of handcrafted acoustic features and Whisper embeddings.

## Approach

The final approach uses two types of features from each audio file.

### 1. Acoustic Features

First, the audio is loaded at 16 kHz using `librosa`.

I extracted 91 handcrafted features from the audio, including:

- MFCC mean and standard deviation
- Delta MFCC mean and standard deviation
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate
- RMS energy
- Audio duration

These features capture different characteristics of the speech such as spectral information, energy and variation in the audio signal.

### 2. Whisper Embeddings

I also used `openai/whisper-small.en` to get a higher-level representation of the spoken audio.

Since the recordings can be around 45–60 seconds long, each audio file is divided into 30-second chunks. The Whisper encoder hidden states are mean-pooled to obtain a 768-dimensional embedding for each audio file.

The acoustic features and Whisper embeddings are then combined into one feature vector.

### 3. Regression Models

I experimented with tree-based regression models and used the following two models in the final pipeline:

- CatBoost Regressor
- ExtraTrees Regressor

The final prediction is an ensemble of both models:

- 59% CatBoost
- 41% ExtraTrees

I also applied a small calibration step to the final predictions based on local validation results. The predictions are clipped to the valid grammar score range of 0 to 5.

## Project Structure

```text
SHL-Grammar-Scoring/
│
├── grammar_scoring.ipynb
├── submission.csv
├── README.md
└── requirements.txt

