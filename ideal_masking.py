import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import savemat

def compute_irm(clean_magnitude, noisy_magnitude):
    """
    Compute the Ideal Ratio Mask (IRM) based on the clean and noisy magnitudes.
    """
    return clean_magnitude / noisy_magnitude

def enhance_speech(clean_file, noisy_file, output_file):
    # Load the clean and noisy audio files
    clean, sr = librosa.load(clean_file, sr=16000)
    noisy, _ = librosa.load(noisy_file, sr=sr)
    
    # Compute STFT for both clean and noisy signals
    clean_stft = librosa.stft(clean, n_fft=320, hop_length=160, window='hann')
    noisy_stft = librosa.stft(noisy, n_fft=320, hop_length=160, window='hann')
    
    # Compute the magnitude spectrograms
    clean_magnitude = np.abs(clean_stft)
    clean_phase = np.angle(clean_stft)
    noisy_magnitude = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)
    
    # Compute the Ideal Ratio Mask (IRM)
    mask = compute_irm(clean_magnitude, noisy_magnitude)
    
    # Apply the mask to the noisy magnitude spectrogram
    enhanced_magnitude = noisy_magnitude * mask
    
    # Reconstruct the STFT of the enhanced speech
    enhanced_stft = enhanced_magnitude * np.exp(1j * clean_phase)

    
    # Compute the inverse STFT to get the enhanced speech signal
    enhanced_speech = librosa.istft(enhanced_stft, n_fft=320, hop_length=160, window='hann', length=len(clean))
    
    f1 = output_file + ".mat"
    f2 = output_file + ".wav"
    # Save the enhanced speech
    savemat(f1, {'mask': mask})
    sf.write(f2, enhanced_speech, samplerate=sr)

    return enhanced_speech

if __name__ == "__main__":
    # Paths to directories
    clean_dir = "/gpfs/home/cdn16hqu/NTCD_TIMIT/clean/test/"
    noisy_dir = "/gpfs/home/cdn16hqu/NTCD_TIMIT/babble/0db/test/wav/"
    output_dir = "./ideal_masks_cp"
    
    for filename in os.listdir(clean_dir):
        if filename.endswith('.wav'):
            clean_file = os.path.join(clean_dir, filename)
            noisy_file = os.path.join(noisy_dir, filename) # Assumes same naming for clean and noisy files
            output_file = os.path.join(output_dir, "enhanced_" + filename[:-4])
            
            enh = enhance_speech(clean_file, noisy_file, output_file)

            speech, sr = librosa.load(clean_file, sr=16000)

            # plt.figure(dpi=400)  # Set the resolution to 400 dpi
            # # Plot the clean signal with a label
            # plt.plot(speech, alpha=0.8, label='Clean')
            # plt.plot(enh, alpha=0.5, label='Ideal mask')
            
            # # Add axis labels
            # plt.xlabel('Time')
            # plt.ylabel('Amplitude')
            # # Add a legend
            # plt.legend()
            # # Save the plot to a file with a higher resolution
            # diff_fname = os.path.join(output_dir, "enhanced_" + filename[:-4] + ".png")
            # plt.savefig(diff_fname, dpi=600)
            # plt.close()
