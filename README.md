# Speech Enhancement Project

This repository contains the Python code for an audio enhancement model using deep learning. The project aims to improve the quality of noisy audio signals using various LSTM-based neural network architectures. The experiments are evaluated on the NTCD_TIMIT dataset, using babble noise at various SNRs.

## Description

The program uses Long Short-Term Memory (LSTM) networks, including vanilla LSTM or an enhanced LSTM model to perform audio signal enhancement. It applies noise masking techniques to process audio and utilizes features like log power spectra (LPS), Mel-frequency cepstral coefficients (MFCCs), and potentially HuBERT representations.

## Installation

Before running this project, you need to install the required dependencies. You can install them using:

```sh
pip install -r requirements.txt
```

Make sure to have the following main dependencies:

- `librosa` for audio processing.
- `torch` for deep learning model construction and training.
- `pypesq` and `pystoi` for evaluation metrics.

## Usage

The project is run from the command line, where you can specify various arguments such as output file directory, batch size, number of epochs, feature type, and learning rate.

```sh
python <script_name>.py [out_file] [batch] [epochs] [feature_type] [lr]
```

Replace `<script_name>` with the actual name of the main script.

### Arguments

- `out_file`: Path to the directory where output files will be saved.
- `batch`: The size of batches for model training.
- `epochs`: The number of epochs for model training.
- `feature_type`: The type of feature used in training (e.g., 'lps', 'mfcc', or 'hubert').
- `lr`: The learning rate for the optimizer.

## Features

- Data loading and preprocessing.
- Feature extraction with support for LPS and MFCC.
- Neural network models for audio enhancement.
- Training and validation split and execution.
- Model training and evaluation with loss plotting.
- Time-domain signal reconstruction and saving enhanced audio files.
- Evaluation using PESQ & ESTOI metrics (commented out; intended for MATLAB use).

## Output

The program outputs enhanced audio files, loss graphs during training, and various plots illustrating the performance and results of the audio enhancement process.


## Acknowledgments

This project makes use of the following open-source libraries:

- [LibROSA](https://librosa.github.io/librosa/)
- [PyTorch](https://pytorch.org/)
- [PESQ](https://github.com/ludlows/python-pesq)
- [STOI](https://github.com/mpariente/pystoi)

We also acknowledge the creators of the datasets and various resources that made this project possible.

## Contact

For any queries or discussions, please open an issue in the repository, and I will get back to you.

```
