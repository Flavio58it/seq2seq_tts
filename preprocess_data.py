"""Data pre-processing and audio feature extraction"""

import argparse
import json
import os

import numpy as np

from audio.Spectrogram import STFTProcessor


def process_data(config, prompts_file, wav_dir, output_dir):
    """Process the data
    """
    text_dir = os.path.join(output_dir, "text")
    feats_dir = os.path.join(output_dir, config["data_processors"]["audio_processor"])

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(feats_dir, exist_ok=True)

    # read data from prompts file
    with open(prompts_file, "r") as fp:
        prompts = fp.readlines()
    prompts = [line.strip("\n") for line in prompts]
    prompts = [line.split() for line in prompts]

    # instantiate audio processor
    if config["data_processors"]["audio_processor"] == "mag" or config["data_processors"]["audio_processor"] == "mel":
        audio_processor = STFTProcessor(sampling_rate=config["audio"]["sampling_rate"], n_fft=config["audio"]["n_fft"],
                                        n_mel=config["audio"]["n_mel"], fmin=config["audio"]["fmin"],
                                        fmax=config["audio"]["fmax"], preemphasis=config["audio"]["preemphasis"],
                                        frame_length=config["audio"]["frame_length"],
                                        frame_shift=config["audio"]["frame_shift"], min_db=config["audio"]["min_db"],
                                        ref_db=config["audio"]["ref_db"])
    else:
        raise NotImplementedError

    for line in prompts:
        # get the filename to be processed
        filename = line[0]

        # extract audio feats
        if config["data_processors"]["audio_processor"] == "mag":
            y = audio_processor.load_wav(os.path.join(wav_dir, filename + ".wav"))
            feats = audio_processor.spectrogram(y).T
        elif config["data_processors"]["audio_processor"] == "mel":
            y = audio_processor.load_wav(os.path.join(wav_dir, filename + ".wav"))
            feats = audio_processor.melspectrogram(y).T
        else:
            raise NotImplementedError

        # number of frames in the extracted features
        num_frames = feats.shape[0]

        # save features of a particular utterance for training only when the number of frames is not too large
        # (to enable the model to learn a clean attention)
        if num_frames <= config["audio"]["max_frames"]:
            print("Processing ... %s ...." % (filename))
            np.save(os.path.join(feats_dir, filename + ".npy"), feats)
            with open(os.path.join(text_dir, filename + ".txt"), "w") as fp:
                fp.write(" ".join(line[1:]))
            fp.close()


if __name__ == "__main__":
    # setup a command line argument parser
    parser = argparse.ArgumentParser(description="Data preprocessing and audio feature extraction")

    parser.add_argument("--config_file", help="Path to the json file containing audio processing settings",
                        required=True)
    parser.add_argument("--prompts_file", help="Path to the file containing the prompts", required=True)
    parser.add_argument("--wav_dir", help="Path to the folder containing the wav files", required=True)
    parser.add_argument("--output_dir", help="Path to the folder where the audio features will be saved to disk",
                        required=True)

    args = parser.parse_args()

    prompts_file = args.prompts_file
    wav_dir = args.wav_dir
    output_dir = args.output_dir

    with open(args.config_file) as fp:
        config = json.load(fp)

    process_data(config, prompts_file, wav_dir, output_dir)
