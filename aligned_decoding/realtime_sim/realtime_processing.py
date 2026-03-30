""" Real-time filtering and processing functions.

Zac Spalding
"""

import scipy.signal as signal
import numpy as np


def process_HG(data, bandpassCoefs, bad_channels=None, filt_ics=None):
    if bad_channels is None:
        bad_channels = []

    # Apply Common Average Reference (CAR) filtering
    data_filtered = CAR(data, bad_channels)

    # Apply bandpass filtering
    filtered_data, filt_ics = filter_HG_bin(data_filtered, bandpassCoefs,
                                            filt_ics)

    # Compute power of the filtered signals
    power = compute_bin_power(filtered_data)

    return power, filt_ics  # Return the power of the filtered signals


def CAR(data, bad_channels=None):
    """Common Average Reference (CAR) filtering."""
    if bad_channels is None:
        bad_channels = []

    # Exclude bad channels
    good_channels = [i for i in range(data.shape[0]) if i not in bad_channels]

    # Compute the average across good channels
    avg = np.mean(data[good_channels, :], axis=0)

    # Subtract the average from each channel
    data_filtered = data - avg

    return data_filtered


def filter_HG_bin(data, bandpassCoefs, band_ics=None):
    # data: (channels, time)
    # bandpassCoefs: (bands, taps, [a, b]) for IIR or (bands, taps) for FIR
    if bandpassCoefs.ndim == 3:  # IIR: (bands, taps, [a, b])
        return IIR_filter_HG_bin(data, bandpassCoefs, band_ics)
    elif bandpassCoefs.ndim == 2:  # FIR: (bands, taps)
        return FIR_filter_HG_bin(data, bandpassCoefs)
    else:
        raise ValueError("bandpassCoefs must be either 2D or 3D array.")


def FIR_filter_HG_bin(data, bandpassCoefs):
    band_signals = []
    for coefs in bandpassCoefs:
        # Apply the filter to the data
        filtered_data = signal.lfilter(coefs, 1.0, data)
        band_signals.append(filtered_data)
    return np.stack(band_signals, axis=-1), None  # (channels, time, bands)


def IIR_filter_HG_bin(data, bandpassCoefs, zi=None):
    n_chan = data.shape[0]

    if zi is None:
        # create initial conditions per channel
        filter_params = []
        for bandCoefs in bandpassCoefs:
            b, a = bandCoefs[:, 1], bandCoefs[:, 0]
            # zi = [signal.lfilter_zi(b, a) for _ in range(n_chan)]
            # zi = np.stack(zi, axis=0)  # (channels, order)
            zi = np.tile(signal.lfilter_zi(b, a), (n_chan, 1))
            filter_params.append((b, a, zi))
    else:
        # Use provided initial conditions
        filter_params = [(bandCoefs[:, 1], bandCoefs[:, 0], zi_band) for
                         bandCoefs, zi_band in zip(bandpassCoefs, zi)]

    band_signals = []
    band_ics = []
    for (b, a, zi) in filter_params:
        filtered_data, zf = signal.lfilter(b, a, data, zi=zi)
        band_signals.append(filtered_data)
        # Update the initial conditions for the next filter
        band_ics.append(zf)
    band_signals = np.stack(band_signals, axis=-1)  # (channels, time, bands)
    band_ics = np.stack(band_ics, axis=0)  # (bands, channels, coefs)
    return band_signals, band_ics


def compute_bin_power(data):
    # data: (channels, time, bands)
    power = np.square(data)
    binned_power = np.mean(power, axis=(1, 2))  # Average across time and bands
    # binned_power = np.mean(power, axis=2)

    # compute RMS power
    binned_power = np.sqrt(binned_power)  # (channels,)

    # binned_power = signal.resample(binned_power, 4, axis=-1)
    return binned_power
