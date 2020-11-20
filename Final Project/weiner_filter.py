# Matthew Dillabough and Zach Litziner
# Final Project

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def generate_noisy_file(fname):
    '''
    Simply function which takes an audio file and adds gaussian white noise to create a
    'noisy' file. Original sound is plotted over new noisy sound and new .wav file is
    written to code base.
    Input:
        - fname: name of audio file
    '''
    freq, data = wavfile.read(fname)
    data = data/32768
    RMS = np.sqrt(np.mean(data**2))

    # Generate gaussian 'white noise' and add to original signal
    noise = np.random.normal(0, RMS, data.shape[0])
    signal = data + noise

    # Plot original signal over new 'noisy' signal
    plt.plot(np.arange(len(signal))/freq, signal)
    plt.plot(np.arange(len(data))/freq, data)
    plt.title('Generated Noisy Signal')
    plt.legend(["Noisy", "Original"])
    plt.show()

    # Write new signal to code base
    outputPath = 'noisy_audio/' + \
        fname.split('/')[1].split('.')[0] + '_noisy.wav'
    wavfile.write(outputPath, freq, signal)


def wiener_filter(audio, freq):
    '''
    Simple wiener filter audio noise reduction filter.
    Input:
        - audio: unfiltered audio file
        - freq: sample rate (Hz)
    Returns:
        - output: filtered audio file
    '''

    audio_length = audio.shape[0]
    # For signals with 16000Hz (standard), multiplying by 0.032 gives us desired 512 frames - could be adjusted
    frame_size = int(np.ceil(0.032 * freq))
    nr_step = 3
    NFFT = 2 * frame_size
    # Hanning function smooths discontinuities at the beginning and end of signal
    han_win = np.hanning(frame_size + 2)[1:-1]

    init_noise = int(0.1 * freq)
    nsum = np.zeros(NFFT)

    for m in range(0, init_noise - frame_size+1, nr_step):
        nwin = audio[m:m+frame_size] * han_win  # Generate FFT for frame
        # Compute energy for frame
        nsum = nsum + np.square(abs(np.fft.fft(nwin, NFFT)))
    bg_noise = nsum / (init_noise - frame_size)  # Average energy per frame

    shift_pct = 0.5
    # Could adjust frame overlap value
    overlap = int(np.ceil((1-shift_pct) * frame_size))
    offset = frame_size - overlap
    max_m = int(np.floor((audio_length - NFFT)/offset)) + 1
    output = np.zeros(audio_length)

    # Could adjust thresholds for filtering signal
    min_SNR = 0.1
    alpha = 0.98

    for m in range(max_m):
        begin = m * offset
        finish = m * offset + frame_size
        s_frame = audio[begin:finish]

        # Apply hanning to frame because frames overlap and don't care so much about edges
        s_frame_win = s_frame * han_win
        # Calculate new FFT for this frame
        s_frame_fft = np.fft.fft(s_frame_win, NFFT)

        # Calculate phase and magnitude
        s_frame_phase = np.angle(s_frame_fft)
        s_frame_mag = abs(s_frame_fft)

        # Calculate signal to noise ratio for given frame
        post_SNR = ((np.square(s_frame_mag)) / bg_noise) - 1
        # Checks if SNR is above threshold, if it is reduces noise to min ratio
        post_SNR[post_SNR < min_SNR] = min_SNR

        # Will adjust magnitudes of frequency bands to minimize ones that are not components of desired signal
        eta = (1 - alpha) * post_SNR
        eta[eta < -19] = -19
        mag_new = (eta / (eta+1)) * s_frame_mag

        # Recombine frequency domain signal mangitude with frequency domain signal phase
        s_frame_fft = mag_new * np.exp(s_frame_phase * 1j)
        # Output generates time domain signal from filtered frequency domain signal
        output[begin:begin + NFFT] = output[begin:begin + NFFT] + \
            (np.fft.ifft(s_frame_fft, NFFT)).real

    return output


def test(fname):
    '''
    Tester for wiener_filter() function which plots original vs. filtered
    audio files.
    Input:
        - fname: Name of audio file to filter
    '''

    # Read audio
    audio_path = fname
    freq, data = wavfile.read(audio_path)

    # Plot unfiltered audio file
    plt.plot(np.arange(len(data))/freq, data)
    plt.title('Noisy vs Filtered Audio')

    # Plot filtered audio file
    output = wiener_filter(data, freq)
    plt.plot(np.arange(len(output))/freq, output)
    plt.legend(["Noisy", "Filtered"])
    plt.show()

    # Create new output audio file with filtered audio data
    outputPath = 'outputs/' + \
        audio_path.split('/')[1].split('.')[0] + '_output.wav'
    wavfile.write(outputPath, freq, output)


def compare(fname):
    '''
    Function which compares the filtered signal of 'noisy' audio to the original signal (without noise)
    via a plot. 
    Input:
        - fname: file name of audio file (i.e. for t1.wav, 't1')
    '''
    originalPath = 'test_audio/' + fname + '.wav'
    oFreq, oData = wavfile.read(originalPath)
    oData = oData/32768

    filteredPath = 'outputs/' + fname + '_noisy_output.wav'
    fFreq, fData = wavfile.read(filteredPath)

    plt.plot(np.arange(len(oData))/oFreq, oData)
    plt.plot(np.arange(len(fData))/fFreq, fData)
    plt.title('Original Signal vs Filtered Signal')
    plt.legend(["Original", "Filtered"])
    plt.show()


if __name__ == '__main__':
    for i in range(1, 11):
        j = str(i)
        compare('t' + j)
