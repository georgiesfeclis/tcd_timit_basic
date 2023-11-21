import librosa.feature
import torch.nn as nn
from utils import *
from train_val import *
from test import *
from dataset import AudioEnhancementDataset
from torch.utils.data import DataLoader, random_split
from model import LSTMModel, LSTMModelAttention, EnhancedLSTMModel
from pypesq import pesq
from pystoi import stoi
import soundfile as sf
import argparse
import time
from scipy.io import savemat

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == '__main__':
    startTime = time.time()

    # Define argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("out_file", type=dir_path, help='Path to outputs')
    parser.add_argument("batch", type=int, default=128, help="Batches")
    parser.add_argument("epochs", type=int, default=30, help="Epochs")
    parser.add_argument("feature_type", type=str, default='lps', help='Feature type used in training')
    parser.add_argument("lr", type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    print(f'Here comes a list of all the arguments used in creating these results')
    print(f'Output files are saved in {args.out_file}')
    print(f'No of batches is  {args.batch} and the no of epochs is {args.epochs}')
    print(f'The feature type used in training is {args.feature_type} and the training target is a magnitude mask.')

    fs = 16000
    feat_type = args.feature_type

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device in use: {device}')

    # 0. DATA LOADING
    clean_train_path = "/gpfs/home/cdn16hqu/NTCD_TIMIT/clean/train/"
    clean_test_path = "/gpfs/home/cdn16hqu/NTCD_TIMIT/clean/test/"

    noisy_train_path = "/gpfs/home/cdn16hqu/NTCD_TIMIT/babble/0db/train/wav/"
    noisy_test_path = "/gpfs/home/cdn16hqu/NTCD_TIMIT/babble/0db/test/wav/"

    # Import clean data
    _, clean_train_spectrogram, _ = load_clean_data(clean_train_path)
    clean_test, clean_test_spectrogram, test_filenames = load_clean_data(clean_test_path)
    # Import noisy data
    noisy_train, noisy_train_spectrogram = load_noisy_data(noisy_train_path)
    noisy_test, noisy_test_spectrogram = load_noisy_data(noisy_test_path)

    print("Data loading finished")

    # 1. DATA PRE-PROCESSING & FEATURE EXTRACTION

    # Compute LPS to get data normalisation info
    if feat_type == 'lps':
        X = compute_log_power_spectra(noisy_train_spectrogram)
        X_test = compute_log_power_spectra(noisy_test_spectrogram)
    elif feat_type =='mfcc':
        X = []
        X_test = []
        for file in noisy_train:
            feat = librosa.feature.mfcc(y=file, sr=fs, n_fft=320, hop_length=160, n_mfcc=60)
            X.append(torch.from_numpy(feat))

        for test_file in noisy_test:
            ft = librosa.feature.mfcc(y=test_file, sr=fs, n_fft=320, hop_length=160, n_mfcc=60)
            X_test.append(torch.from_numpy(ft))
    elif feat_type == 'hubert':
        # Add something here to load huberts
        print("No support for huberts yet")
    else:
        raise TypeError("The feature type selected does not exist!")

    # Flatten each tensor to a 1D tensor before concatenating
    flattened_features = [tensor.reshape(-1) for tensor in X]

    # Calculate global mean and std
    global_mean = torch.mean(torch.cat(flattened_features))
    # global_mean = 0
    global_std = torch.std(torch.cat(flattened_features))
    # global_std = 1
    global_pad_size = max(max([feature.shape[1] for feature in X]), max([feat.shape[1] for feat in X_test]))

    print(f'The mean feature values in {global_mean} and the std is {global_std}')

    # 2. DATA PREPARATION - TRAIN / VALIDATION SPLIT

    # Then you can feed these into your dataset
    dataset = AudioEnhancementDataset(X, noisy_train_spectrogram, clean_train_spectrogram, global_mean,
                                    global_std, global_pad_size)
    # Set the random seed for reproducibility
    torch.manual_seed(10)
    train_set_size = int(len(dataset) * 0.7)
    valid_set_size = len(dataset) - train_set_size

    print(train_set_size, valid_set_size)
    trainLoader, validLoader = random_split(dataset, [train_set_size, valid_set_size])


    trainLoader = DataLoader(trainLoader,
                            batch_size=args.batch,
                            shuffle=True)
    validLoader = DataLoader(validLoader,
                            batch_size=args.batch,
                            shuffle=True)
    # DEBUGGING
    # Iterate over the DataLoader and print the shapes of batches
    for i, (batch_data, batch_labels, _) in enumerate(trainLoader):
        print(f"Batch {i + 1}")
        print("Data shape:", batch_data.shape)
        print("Labels shape:", batch_labels.shape)

        # plt.figure()
        # plt.subplot(211)
        # plt.imshow(batch_data[0].T, origin='lower')
        # plt.colorbar()
        # plt.subplot(212)
        # plt.imshow(batch_labels[0].T, origin='lower')
        # plt.colorbar()
        # plt.show()

        # For demonstration, let's just check the first 3 batches
        if i == 2:
            break

    # 3. MODEL INITIALISATION
    third_dim_size = len(X[0])
    # model = LSTMModelAttention(third_dim_size, global_pad_size).to(device)
    # model = EnhancedLSTMModel(third_dim_size, global_pad_size).to(device)
    model = LSTMModel(third_dim_size, global_pad_size).to(device)

    # Define a loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Keep track of loss
    training_loss = []
    validation_loss = []


    # 4. MODEL TRAINING
    for epoch in range(args.epochs):
        training_loss = train(model, device, lossFunction=loss_function, lossList=training_loss, trainLoader=trainLoader,
            optimizer=optimizer, epoch=epoch)
        validation_loss = validate(model, device, lossFunction=loss_function, lossList=validation_loss, validLoader=validLoader,
                epoch=epoch)
        scheduler.step()  # Decrease the learning rate every 30 epochs
    
    # Save model weights
    weights_fname = str(args.out_file) + '/model.pth'
    torch.save(model.state_dict(), weights_fname)

    # Plot loss
    plt.figure()
    plt.plot(np.array(training_loss), 'r')
    plt.plot(np.array(validation_loss))
    loss_fname = str(args.out_file) + '/loss.png'
    print(f'Loss graph is saved in {loss_fname}')
    plt.savefig(loss_fname)

    # 5. MODEL EVALUATION & MASK ESTIMATION
    # Set up the test dataset & data loader
    test_set = AudioEnhancementDataset(X_test, noisy_test_spectrogram, clean_test_spectrogram, global_mean,
                                    global_std, global_pad_size)
    testLoader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False)

    # Evaluate model:
    mask_test = []
    # Evaluate model
    with torch.no_grad():
        masks = test(model, device, testLoader)
        # print("Masks shape in main after eval: ", np.shape(masks))

    # 6. MASK SPEECH AND RECONSTRUCT TIME-DOMAIN AUDIO
    # Mask speech
    maskedSpeech = mask_speech(noisy_test, clean_test, masks)
    # idealMaskedSpeech = mask_speech(clean_test, noisy_test_spectrogram, clean_test, idealMasks)

    # 7. EVALUATE PESQ & ESTOI FOR ENHANCED SPEECH
    # NOT DOING THIS BECAUSE WE ARE DOING EVALUATION IN MATLAB INSTEAD. 
    # pesqList = []
    # stoiList = []

    # pesqListI = []
    # stoiListI = []
    # for clean, enhanced in zip(clean_test, maskedSpeech):
    # # for clean, enhanced, maskEnh in zip(clean_test, maskedSpeech, idealMaskedSpeech):

    #     assert np.shape(clean) == np.shape(enhanced)

    #     pesq_val = pesq(clean, enhanced, fs=fs)
    #     pesqList.append(pesq_val)

    #     stoi_val = stoi(clean, enhanced, fs_sig=16000, extended=True)
    #     stoiList.append(stoi_val)

        # Ipesq_val = pesq(clean, maskEnh, fs=fs)
        # pesqListI.append(Ipesq_val)

        # istoi_val = stoi(clean, maskEnh, fs_sig=16000, extended=True)
        # stoiListI.append(istoi_val)

    # Assuming clean_test[10] and maskedSpeech[10] contain your audio signals
    # plt.figure(dpi=400)  # Set the resolution to 400 dpi
    # # Plot the clean signal with a label
    # plt.plot(clean_test[76], alpha=0.8, label='Clean')
    # # plt.plot(idealMaskedSpeech[20], alpha=0.7, label='Ideal mask')
    # # Plot the enhanced signal with a label
    # plt.plot(maskedSpeech[76], alpha=0.8, label='Enhanced')
    
    # # Add axis labels
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # # Add a legend
    # plt.legend()
    # # Save the plot to a file with a higher resolution
    # diff_fname = str(args.out_file) + '/difference.png'
    # plt.savefig(diff_fname, dpi=400)


    # print("Mean PESQ Score: ", np.mean(pesqList))
    # print("Mean ESTOI Score: ", np.mean(stoiList))

    # print("Mean ideal PESQ Score: ", np.mean(pesqListI))
    # print("Mean ideal ESTOI Score: ", np.mean(stoiListI))

    

    # 8. SAVE TIME-DOMAIN SIGNALS, PLOTS AND MASKS IN OUTPUT FILES
    print(f'Finished evaluation and masking. Start saving audio files and images in {args.out_file} ...')
    print(f'No of masked files is {len(maskedSpeech)}')
    for i in range(0, len(maskedSpeech)):
    # for i in range(0, 15):
        clean_stft = librosa.stft(clean_test[i], n_fft=320, hop_length=160, window='hann')

        filename = str(args.out_file) + '/enhanced/' + test_filenames[i] + '.wav'
        sf.write(filename, maskedSpeech[i], samplerate=fs)

        if i % 100 == 0:
            fname = str(args.out_file) + '/img_outputs/' + test_filenames[i] + '.png'
            plt.figure(figsize=(10,25), dpi=400)  # Set the resolution to 400 dpi

            # First subplot: Clean Spectrogram
            # First subplot: Clean Spectrogram
            plt.subplot(411)
            plt.imshow(np.log(np.abs(clean_stft)), aspect='auto', origin='lower')
            plt.colorbar()
            plt.ylim(0, 161)  # Set y-axis limit to 0-161
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency Bins')
            plt.title('Clean Spectrogram for ' + str(test_filenames[i]))

            # Second subplot: Noisy Spectrogram
            plt.subplot(412)
            plt.imshow(np.log(noisy_test_spectrogram[i]), aspect='auto', origin='lower')
            plt.colorbar()
            plt.ylim(0, 161)  # Set y-axis limit to 0-161
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency Bins')
            plt.title('Noisy Spectrogram')

            # Third subplot: Mask
            plt.subplot(413)
            plt.imshow(masks[i], aspect='auto', origin='lower')
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


   # Get duration of program
    elapsed = time.time() - startTime
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed)))


