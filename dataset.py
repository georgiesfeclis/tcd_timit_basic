import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class AudioEnhancementDataset(Dataset):
    def __init__(self, noisy_features, noisy_spectrograms, clean_spectrograms, global_mean, global_std, global_pad_size):
        # Assert lengths equal
        assert len(noisy_features) == len(noisy_spectrograms) == len(clean_spectrograms)
        # Normalize and convert features to tensors
        self.noisy_tensors = [(feature.T.clone().detach() - global_mean) / global_std for feature in noisy_features]

        # Store the original lengths of the features and labels
        self.original_feature_lengths = [feature.shape[0] for feature in self.noisy_tensors]

        # Compute magnitude masks for labels & force the highest value = 1 using clamp
        self.labels = [torch.tensor(clean_spectrogram.T / (noisy_spectrogram.T + 1e-8)).clamp(min=0, max=1) for clean_spectrogram, noisy_spectrogram in zip(clean_spectrograms, noisy_spectrograms)]

        # Validate that lengths of features are equal to the lengths of the labels
        if self.original_feature_lengths != [label.shape[0] for label in self.labels]:
            raise ValueError("Lengths of features are not equal to the lengths of the labels!")

        # Pad the features and labels to global_pad_size
        self.padded_noisy_tensors = [F.pad(tensor, (0, 0, 0, global_pad_size - tensor.shape[0])) for tensor in self.noisy_tensors]
        self.padded_labels = [F.pad(label, (0, 0, 0, global_pad_size - label.shape[0])) for label in self.labels]

    def __len__(self):
        return len(self.noisy_tensors)

    def __getitem__(self, idx):
        return self.padded_noisy_tensors[idx], self.padded_labels[idx], self.original_feature_lengths[idx]

