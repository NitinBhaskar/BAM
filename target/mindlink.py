import numpy as np
#from scipy.signal import butter, lfilter

# Function to read one byte from serial
def ReadOneByte(ser):
    return int.from_bytes(ser.read(1), byteorder='little')

# Taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
#def butter_bandpass(lowcut, highcut, fs, order=5):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype='band')
#    return b, a

#Taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y

def read_raw_eeg(ser, nsamples):
    sample_count = 0
    samples = []

    # Sample rate
    fs = 512.0

    # Need filtering from 2Hz to 35Hz
    lowcut = 2.0
    highcut = 35.0
    order = 6

    while sample_count < nsamples:
        if ReadOneByte(ser) == 0xAA:  # Sync 1
            if ReadOneByte(ser) == 0xAA:  # Sync 2
                plen = ReadOneByte(ser)  # Packet length
                if plen < 169:
                    if ReadOneByte(ser) == 0x80:  # Check for RAW data
                        if ReadOneByte(ser) == 0x02:  # Check the Raw data len
                            raw = ReadOneByte(ser) * 256  # Read higher 8-bits
                            raw += ReadOneByte(ser)  # Read lower 8-bits
                            if raw > 32768:
                                raw -= 65536
                            samples.append(raw)
                            sample_count += 1
                    else:
                        for i in range(plen):  # Other than raw data
                            ReadOneByte(ser)

    # Compute mean and standard deviation
    mean, std = np.mean(samples), np.std(samples)
    samples = samples - mean
    samples = samples/std

    # Filter raw data
#    filtered_samples = butter_bandpass_filter(samples, lowcut, highcut, fs, order=6)
    return samples
