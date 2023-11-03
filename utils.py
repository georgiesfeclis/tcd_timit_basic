import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt


def load_noisy_data(noisy_dir, n_fft=320, hop_length=160):
    noisy_data = []
    noisy_spectrograms = []

    # Iterate over all files in the noisy data directory
    for dir_path, dir_names, filenames in os.walk(noisy_dir):
        for filename in sorted(filenames):
            if filename.endswith('.wav'):
                # Read the WAV file
                signal, sample_rate = librosa.load(os.path.join(dir_path, filename), sr=None)
                noisy_data.append(signal)

                # Calculate the spectrogram
                stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window='hann')
                spectrogram = np.abs(stft_result)
                noisy_spectrograms.append(spectrogram)

    return noisy_data, noisy_spectrograms 


def load_clean_data(clean_dir, n_fft=320, hop_length=160):
    clean_data = []
    clean_spectrograms = []
    filename_list = []
    # Iterate over all files in the clean data directory
    for dir_path, dir_names, filenames in os.walk(clean_dir):
        for filename in sorted(filenames):
            if filename.endswith('.wav'):
                # Read the WAV file
                signal, sample_rate = librosa.load(os.path.join(dir_path, filename), sr=None)
                clean_data.append(signal)

                # Calculate the spectrogram
                stft_result = librosa.stft(signal, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, window='hann')
                spectrogram = np.abs(stft_result)
                clean_spectrograms.append(spectrogram)

                cleanFile = filename.replace(dir_path, '')
                cleanFile = cleanFile.replace('.wav', '')
                filename_list.append(cleanFile)

    return clean_data, clean_spectrograms, filename_list


# Compute the log power spectra for noisy data
def compute_log_power_spectra(spectrograms):
    return [torch.log(torch.Tensor(torch.from_numpy(spectrogram) ** 2)) for spectrogram in spectrograms]


# Reconstruct time-domain speech
def mask_speech(noisy, clean, masks, n_fft=320, hop_length=160):
    # Check the number of test spectrograms matches the number of masks
    assert len(noisy) == len(masks)
    # print("noisy ", len(noisy))
    # print("masks ", len(masks))

    # Define output list
    enhancedList = []
    for i in range(0, len(masks)):
        # Check the length of each mask matches the length of each spectrogram
        clean_stft = librosa.stft(np.array(clean[i]), n_fft=320, win_length=320, hop_length=160, window='hann')
        noisy_stft = librosa.stft(np.array(noisy[i]), n_fft=320, win_length=320, hop_length=160, window='hann')

        noisy_mag = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)

        # print(f'shape of mask: {np.shape(masks[i])}')
        # print(f'shape of noisy magSpec: {np.shape(noisy_mag)}')
        # print(f'shape of noisy td: {np.shape(noisy[i])}')
        # print(f'shape of noisy stft: {np.shape(noisy_stft)}')


        assert len(noisy_mag) == len(masks[i])

        # Mask the magnitude
        masked = np.multiply(noisy_mag, masks[i])

        # Reconstruct complex enhanced speech
        complex_stft = masked * np.exp(1j * noisy_phase)
        # print(np.shape(complex))
        # print(np.shape(noisy_spectrograms[i]))
        # Reconstruct time-domain audio:
        stft_result = librosa.istft(complex_stft, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, window='hann', length=len(clean[i]))
        # plt.figure()
        # plt.subplot(311)
        # plt.imshow(noisy_magnitude, origin='lower')
        # plt.subplot(312)
        # plt.imshow(masks[i], origin='lower')
        # plt.subplot(313)
        # plt.imshow(masked, origin='lower')
        # plt.show()
        # Check length of reconstructed speech matches the noisy time-domain speech
        assert len(clean[i]) == len(stft_result)

        # Create output list
        enhancedList.append(stft_result)

    return enhancedList

