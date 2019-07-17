"""Script to perform copy synthesis (to test the audio processing code)"""

import argparse
import json

from audio.Spectrogram import STFTProcessor


def do_copy_synthesis(config, wav_file, output_path):
    """Perform copy synthesis on the wav file
    """
    if config["audio_processor"] == "mel" or config["audio_processor"] == "mag":
        audio_processor = STFTProcessor(sampling_rate=config["sampling_rate"], n_fft=config["n_fft"],
                                        n_mel=config["n_mel"], fmin=config["fmin"], fmax=config["fmax"],
                                        preemphasis=config["preemphasis"], frame_length=config["frame_length"],
                                        frame_shift=config["frame_shift"], min_db=config["min_db"],
                                        ref_db=config["ref_db"])
    else:
        raise NotImplementedError

    # read the wav file and get the signal
    y = audio_processor.load_wav(wav_file)

    # compute log-magnitude / mel spectrograms from signal
    mag = audio_processor.spectrogram(y)
    mel = audio_processor.melspectrogram(y)

    # reconstruct the signal from spectrograms
    y_hat_mag = audio_processor.inv_spectrogram(mag, num_gl_iters=60)
    y_hat_mel = audio_processor.inv_melspectrogram(mel, num_gl_iters=60)

    # write reconstructed signals to wav files on disk
    audio_processor.save_wav(y_hat_mag, "%s/%s" % (output_path, "mag_" + wav_file))
    audio_processor.save_wav(y_hat_mel, "%s/%s" % (output_path, "mel_" + wav_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy synthesis")

    parser.add_argument("--config_file", help="Path to the config file containing the audio parameters", required=True)
    parser.add_argument("--wav_file", help="Path to the wav file on which to perform copy synthesis", required=True)
    parser.add_argument("--output_path", help="Path to location where the reconstructed wav files will be written \
                        to disk", required=True)

    args = parser.parse_args()

    config_file = args.config_file
    wav_file = args.wav_file
    output_path = args.output_path

    # read the config parameters from the file
    with open(config_file) as fp:
        config = json.load(fp)

    do_copy_synthesis(config, wav_file, output_path)
