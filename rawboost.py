"""Original RawBoost operators and native-rate wrapper.
MIT License

Copyright (c) 2021 Hemlata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy


def randRange(x1, x2, integer):
    y = float(np.random.uniform(low=x1, high=x2))
    if integer:
        y = int(y)
    return y


def normWav(x, always):
    if always:
        x = x / max(float(np.amax(abs(x))), 1e-12)
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(
    x,
    N_f,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    minBiasLinNonLin,
    maxBiasLinNonLin,
    fs,
):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1)
    )
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y


# Stationary signal independent noise


def SSI_additive_noise(
    x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    x = x + noise
    return x


def rawboost(waveform, sample_rate, algorithm=5):
    x = np.asarray(waveform, dtype=np.float64).copy()
    if sample_rate not in (16000, 48000):
        raise ValueError(sample_rate)
    if algorithm not in (4, 5):
        raise ValueError(algorithm)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite RawBoost input")
    if np.max(np.abs(x)) < 1e-12:
        return x.astype(np.float32)
    # Keep the original frequency parameters in Hz for BOTH sampling rates.
    x = LnL_convolutive_noise(x, 5, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, 5, 20, sample_rate)
    x = ISD_additive_noise(x, 10, 2)
    if algorithm == 4:
        x = SSI_additive_noise(x, 10, 40, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, sample_rate)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite RawBoost output")
    return x.astype(np.float32)
