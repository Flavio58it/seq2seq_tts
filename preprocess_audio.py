"""Audio processing and feature extraction"""

import argparse
import json
import os

import numpy as np

from audio.Spectrogram import STFTProcessor


def feature_extraction(wav_dir, output_dir, config):
    """Process audio files and perform feature extraction
    """
    output_dir = os.path.join(output_dir, config["audio_processor"])
    os.makedirs(output_dir, exist_ok=True)

    if config["audio_processor"] == "mag" or config["audio_processor"] == "mel":
        audio_processor = STFTProcessor(sampling_rate=config["sampling_rate"], n_fft=config["n_fft"],
                                        n_mels=config["n_mels"], preemphasis=config["preemphasis"],
                                        frame_length=config["frame_length"], frame_shift=config["frame_shift"],
                                        min_db=config["min_db"], ref_db=config["ref_db"], fmin=config["fmin"],
                                        fmax=config["fmax"])
    else:
        raise NotImplementedError

    for wavfile in os.listdir(wav_dir):
        # get the filename
        filename = os.path.splitext(wavfile)[0]
        print("Processing ..... %s ....." % (filename))

        if config["audio_processor"] == "mel":
            # read the wav file into a signal
            y = audio_processor.load_wav(os.path.join(wav_dir, wavfile))

            # extract features from signal
            feats = audio_processor.mel_spectrogram(y).T
        elif config["audio_processor"] == "mag":
            # read the wav file into a signal
            y = audio_processor.load_wav(os.path.join(wav_dir, wavfile))

            # extract features from signal
            feats = audio_processor.spectrogram(y).T
        else:
            raise NotImplementedError

        # write the features to disk
        np.save(os.path.join(output_dir, filename + ".npy"), feats)


if __name__ == "__main__":
    # setup a command line argument parser
    parser = argparse.ArgumentParser(description="Audio processing and feature extraction")

    parser.add_argument("--config_file", help="Path to the json file containing audio processing settings",
                        required=True)
    parser.add_argument("--wav_dir", help="Path to the folder containing the wav files", required=True)
    parser.add_argument("--output_dir", help="Path to the folder where the audio features will be saved to disk",
                        required=True)

    args = parser.parse_args()

    wav_dir = args.wav_dir
    output_dir = args.output_dir

    with open(args.config_file) as fp:
        config = json.load(fp)

    feature_extraction(wav_dir, output_dir, config)
