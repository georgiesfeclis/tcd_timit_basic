import os
import torch
import numpy as np
import librosa
from utils import *
# Assuming your model's architecture is defined in `your_model_file.py`
from model import LSTMModel
from dataset import AudioEnhancementDataset
from torch.utils.data import DataLoader
from test import *
from scipy.io import savemat
import soundfile as sf

fs=16000
batch=1
device = torch.device(
        'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f'Device in use: {device}')
global_pad_size=822
global_mean = -6.767367362976074
global_std = 3.528606414794922


# Define paths for clean and noisy audio folders
clean_folder = "/gpfs/home/cdn16hqu/NTCD_TIMIT/clean/test/"
noisy_folder = "/gpfs/home/cdn16hqu/NTCD_TIMIT/babble/0db/test/wav/"

clean_test, clean_test_spectrogram, test_filenames = load_clean_data(clean_folder)
noisy_test, noisy_test_spectrogram = load_noisy_data(noisy_folder)
X_test = compute_log_power_spectra(noisy_test_spectrogram)

test_set = AudioEnhancementDataset(X_test, noisy_test_spectrogram, clean_test_spectrogram, global_mean,
                                    global_std, global_pad_size)
testLoader = DataLoader(test_set,
                        batch_size=batch,
                        shuffle=False)

# Load the model and its weights
third_dim_size = len(X_test[0])
model = LSTMModel(third_dim_size, global_pad_size).to(device)
model.load_state_dict(torch.load("/gpfs/home/cdn16hqu/tcd_timit_basic/results/tcd_lstm_16077984/model.pth"))

# Evaluate model
with torch.no_grad():
    masks = test(model, device, testLoader)

maskedSpeech = mask_speech(noisy_test, clean_test, masks)

outfile = '/gpfs/home/cdn16hqu/tcd_timit_basic/results/tcd_lstm_16077984'

# # Save the reconstructed audio (adding a prefix to indicate it's reconstructed)
# output_path = os.path.join('output_folder_path', 'reconstructed_' + filename)
# librosa.output.write_wav(output_path, reconstructed_signal, sr)

for i in range(0, len(maskedSpeech)):
    # for i in range(0, 15):
        clean_stft = librosa.stft(clean_test[i], n_fft=320, hop_length=160, window='hann')

        filename = outfile + '/enhanced/' + test_filenames[i] + '.wav'
        # filename2 = str(args.out_file) + '/enhanced/' + test_filenames[i] + '_ideal.wav'

        # sf.write(filename, maskedSpeech[i], samplerate=fs)
        sf.write(filename, maskedSpeech[i], samplerate=fs)

        # f1 = outfile + "/masks/" + test_filenames[i] + ".mat"
        # # Save the enhanced speech
        # savemat(f1, {'mask': masks[i]})



        if i < 10:
            fname = outfile + '/img_outputs/' + test_filenames[i] + '.png'
            plt.figure(dpi=400)  # Set the resolution to 400 dpi

            # First subplot: Clean Spectrogram
            plt.subplot(411)
            plt.imshow(np.log(np.abs(noisy_test_spectrogram[i])), origin='lower')
            plt.colorbar()
            plt.ylim(0, 161)  # Set y-axis limit to 0-161
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency Bins')
            plt.title('Clean Spectrogram')

            # Second subplot: Noisy Spectrogram
            plt.subplot(412)
            plt.imshow(np.log(np.abs(noisy_test_spectrogram[i])), origin='lower')
            plt.colorbar()
            plt.ylim(0, 161)  # Set y-axis limit to 0-161
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency Bins')
            plt.title('Noisy Spectrogram')

            # Third subplot: Mask
            plt.subplot(413)
            plt.imshow(masks[i], origin='lower')
            plt.colorbar()
            plt.ylim(0, 161)  # Set y-axis limit to 0-161
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency Bins')
            plt.title('Predicted Mask')

            # Fourth subplot: Time-domain comparison
            plt.subplot(414)
            plt.plot(clean_test[i], alpha=0.8, label='Clean')
            plt.plot(maskedSpeech[i], alpha=0.8, label='Enhanced')
            plt.legend()
            plt.title('Time-domain comparison')

            # Adjust spacing
            plt.tight_layout()

            # Save the plot to a file with a higher resolution
            plt.savefig(fname, dpi=400)
            plt.close()
