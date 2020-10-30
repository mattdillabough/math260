# Matt Dillabough - 10/26/20
# Discussed some of this assignment with Adi Pall

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from numpy import pi, sin
from scipy.io import wavfile
from scipy.signal import find_peaks


def tone_data():
    """ Builds the data for the phone number sounds...
        Returns:
            tones - list of the freqs. present in the phone number sounds
            nums - a dictionary mapping the num. k to its two freqs.
            pairs - a dictionary mapping the two freqs. to the nums
        Each number is represented by a pair of frequencies: a 'low' and 'high'
        For example, 4 is represented by 697 (low), 1336 (high),
        so nums[4] = (697, 1336)
        and pairs[(697, 1336)] = 4
    """
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]  # (Hz)

    nums = {}
    for k in range(0, 3):
        nums[k+1] = (lows[k], highs[0])
        nums[k+4] = (lows[k], highs[1])
        nums[k+7] = (lows[k], highs[2])
    nums[0] = (lows[1], highs[3])

    pairs = {}
    for k, v in nums.items():
        pairs[(v[0], v[1])] = k

    tones = lows + highs  # combine to get total list of freqs.
    return tones, nums, pairs


def load_wav(fname):
    """ Loads a .wav file, returning the sound data.
        If stereo, converts to mono by averaging the two channels
        Returns:
            rate - the sample rate (in samples/sec)
            data - an np.array (1d) of the samples.
            length - the duration of the sound (sec)
    """
    rate, data = wavfile.read(fname)
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:, 0] + data[:, 1]  # stereo -> mono
    length = data.shape[0] / rate
    print(f"Loaded sound file {fname}.")
    return rate, data, length


def dft():
    ''' Loads data from "0.wav" file and then generates DFT real and imaginary plots
        based on the soundwave data
    '''
    rate, data, length = load_wav("noisy_dial.wav")

    freq = fft.fftfreq(data.shape[0], 1/rate)
    sf = fft.fft(data)/data.shape[0]

    dualplot(freq, sf, "DFT (real & imag parts)")
    plt.show()


def analyzePeaks(peaks):
    ''' Takes a tuple of two peaks and then compares to num frequencies to return a digit
    '''
    tones, nums, pairs = tone_data()
    digit = -1
    closestDiff = 100000

    for num in nums:
        interior = (nums[num][0] - peaks[1])**2 + \
            (nums[num][1] - peaks[0])**2
        diff = np.sqrt(interior)
        if diff < closestDiff and diff < 100:
            digit = num
            closestDiff = diff

    if digit != -1:  # some kind of success condition...
        return str(digit)
    else:
        return "X"


def genPeaks(data, rate):
    ''' Simply takes an array of frequencies and a rate, generates the FFTs and then 
        returns the correct peaks within that array
    '''
    freq = fft.fftfreq(data.shape[0], 1/rate)
    freq = fft.fftshift(freq)
    transform = fft.fft(data)
    transform = fft.fftshift(transform)
    abs_transform = np.abs(transform)

    # If you change prom to 10e5, it correctly computes for noisy dial
    peaks = find_peaks(abs_transform, prominence=10e6)
    actual_peaks = peaks[0][0:2]
    actual_peaks = (int(round(abs(freq[actual_peaks[0]]))), int(
        round(abs(freq[actual_peaks[1]]))))

    return actual_peaks


def identify_digit(fname):
    ''' Takes in a soundwave file then generates a DFT plot of the take
        Uses this data to find the low/high peaks in the soundwave then
        identifies and returns the digit associated with that specific tone
    '''
    tones, nums, pairs = tone_data()
    rate, data, length = load_wav(fname)

    actual_peaks = genPeaks(data, rate)

    return analyzePeaks(actual_peaks)


def identify_dial(fname):
    ''' Divides the dial wav file into 7 arrays (each of length 5735 - this value works to split up 
        the separate dials) and then returns the phone number played in the sound
    '''
    tone_length = 0.7  # signal broken into 0.7 sec chunks with one num each
    rate, data, sound_length = load_wav(fname)

    # Hardcoded 5735 as it correctly splits the frequencies into the separate digit dials
    chunks = [data[i:i + 5735] for i in range(0, len(data), 5735)]
    digits = ''

    for chunk in chunks:
        actual_peaks = genPeaks(chunk, rate)
        digits += analyzePeaks(actual_peaks)

    return digits


def dualplot(freq, sf, name):
    """ simple plot of real and imaginary parts """
    plt.figure(figsize=(6.5, 2.5))
    plt.suptitle(name)
    plt.subplot(1, 2, 1)
    plt.loglog(freq, np.real(sf), '.k')
    plt.ylabel('Re(F)')
    plt.subplot(1, 2, 2)
    plt.loglog(freq, np.imag(sf), '.k')
    plt.ylabel('Im(F)')
    plt.subplots_adjust(wspace=0.5)


if __name__ == "__main__":
    # dft()
    print(identify_digit("0.wav"))
    print(identify_digit("5.wav"))
    print(identify_dial("dial.wav"))
    print(identify_dial("dial2.wav"))
    # print(identify_dial("noisy_dial.wav")) If you change prominence of find_peak to 10e5, identify_dial function works
