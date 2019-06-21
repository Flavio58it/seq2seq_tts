"""Audio processing using Spectrograms"""

import librosa
import numpy as np
from scipy import signal, io


class STFTProcessor():
    """Audio processing class using STFT
    """
    def __init__(self, sampling_rate, n_fft, preemphasis, frame_length, frame_shift, min_db, ref_db):
        """Constructor
        """
        self.sampling_rate = sampling_rate
        self.min_db = min_db
        self.ref_db = ref_db
        self.n_fft = n_fft
        self.pre = preemphasis
        self.hop_length = int(frame_shift * sampling_rate)
        self.win_length = int(frame_length * sampling_rate)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_db / 20 * np.log(10))

        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _preemphasis(self, x):
        return signal.lfilter([1, -self.pre], [1], x)

    def _inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.pre], x)

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)

    def _trim_silence(self, y):
        margin = int(self.sampling_rate * 0.1)
        y = y[margin:-margin]

        return librosa.effects.trim(y, top_db=40, frame_length=1024, hop_length=256)[0]

    def _griffin_lim(self, S, num_gl_iters):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(num_gl_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)

        return y

    def _normalize(self, S):
        S_norm = ((S - self.min_db) / - self.min_db)
        S_norm = np.clip(S_norm, 0, 1)

        return S_norm

    def _denormalize(self, S):
        S = np.clip(S, 0, 1)
        S_denorm = (S * -self.min_db) + self.min_db

        return S_denorm

    def load_wav(self, wavpath):
        """Load the wav file given by wavpath from disk and return the signal
        """
        y, sr = librosa.load(wavpath, sr=self.sampling_rate)
        # y = self._trim_silence(y)

        assert self.sampling_rate == sr

        return y

    def save_wav(self, y, path):
        """Save the signal to disk as wav
        """
        y_norm = y * (32767 / max(0.01, np.max(np.abs(y))))
        io.wavfile.write(path, self.sampling_rate, y_norm.astype(np.int16))

    def spectrogram(self, y):
        """Compute log-magnitude spectrogram of a signal
        """
        D = self._stft(self._preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.ref_db

        return self._normalize(S)

    def inv_spectrogram(self, spectrogram, num_gl_iters):
        """Invert spectrogram back to signal
        """
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_db)

        return self._inv_preemphasis(self._griffin_lim(S**1.5, num_gl_iters))
