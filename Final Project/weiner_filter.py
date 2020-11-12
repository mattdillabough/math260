# Matthew Dillabough and Zach Litziner
# Final Project

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def round_up(x):
    return int(x+0.5)


def wiener_filter(audio, freq):
    '''
    Simple wiener filter audio noise reduction filter
    Input:
        - audio: unfiltered audio file
        - freq: frequency of audio file (Hz)
    Returaudio:
        - output: filtered audio file data
    '''

    audio_length = audio.shape[0]
    frame_size = round_up(0.032 * freq)
    nr_step = 3
    NFFT = 2 * frame_size
    han_win = np.hanning(frame_size + 2)[1:-1]

    init_noise = int(0.1 * freq)
    nsum = np.zeros(NFFT)

    for m in range(0, init_noise-frame_size+1, nr_step):
        nwin = audio[m:m+frame_size] * han_win
        nsum = nsum + np.square(abs(np.fft.fft(nwin, NFFT)))
    bg_noise = nsum/(init_noise - frame_size)

    shift_pct = 0.5
    overlap = round_up((1-shift_pct) * frame_size)
    offset = frame_size - overlap
    max_m = int(np.floor((audio_length - NFFT)/offset)) + 1
    mag_old = np.zeros(NFFT)
    output = np.zeros(audio_length)

    min_SNR = 0.1
    alpha = 0.98

    for m in range(max_m):
        begin = m * offset
        finish = m * offset + frame_size
        s_frame = audio[begin:finish]

        s_frame_win = s_frame * han_win
        s_frame_fft = np.fft.fft(s_frame_win, NFFT)

        s_frame_phase = np.angle(s_frame_fft)
        s_frame_mag = abs(s_frame_fft)

        post_SNR = ((np.square(s_frame_mag)) / bg_noise) - 1
        post_SNR[post_SNR < min_SNR] = min_SNR

        eta = alpha*(np.square(mag_old)/bg_noise) + (1 - alpha) * post_SNR
        eta[eta < -19] = -19
        mag_new = (eta / (eta+1)) * s_frame_mag

        s_frame_fft = mag_new * np.exp(s_frame_phase * 1j)
        output[begin:begin + NFFT] = output[begin:begin + NFFT] + \
            (np.fft.ifft(s_frame_fft, NFFT)).real

    return output


# Read audio
audio_path = 'test1.wav'
freq, data = wavfile.read(audio_path)
data = data/32768  # Dividing by 32768 adjusts range of data to be [-1, 1]

# Plot unfiltered audio file
plt.plot(np.arange(len(data))/freq, data)
plt.title('Original Audio')
plt.show()

# Plot filtered audio file
output = wiener_filter(data, freq)
plt.plot(np.arange(len(output))/freq, output)
plt.title('Filtered Audio')
plt.show()

# Create new output audio file with filtered audio data
wavfile.write('output.wav', freq, output)
