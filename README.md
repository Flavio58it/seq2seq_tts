This repository contains code for building text-to-speech synthesizers using seq2seq architectures in conjunction with attention models. 
<br />

The text is parameterized as a sequence of character embeddings, while the corresponding speech is represented by log-magnitude spectrograms.
<br />

The model is a seq2seq feature prediction network with attention which predicts a sequence of spectrogram frames from an input character sequence. The waveforms are then synthesized from the predicted spectrogram frames using the Griffin-Lim algorithm for phase estimation followed by an ISTFT.
<br />

Requirements: <br />
        (1) python 3.7 <br />
        (2) tensorflow 2.0 <br />
        (3) librosa <br />

### Completed Work

### Ongoing Work
- [x] Tacotron2 type model (https://arxiv.org/pdf/1712.05884.pdf)

### Future Work
- [ ] Add support for vocoder parameters (for e.g. WORLD / STRAIGHT etc.)
- [ ] Implement other attention based seq2seq prediction models like Transformer (https://arxiv.org/pdf/1706.03762.pdf)
- [ ] Neural vocoding (wavenet/waveRNN) to generate waveform samples directly conditioned on any predicted
     acoustic representation (spectrograms / vocoder parameters etc...)
- [ ] Add support for global style tokens / prosody embeddings for prosody modeling and transfer (as in
     https://arxiv.org/pdf/1803.09017.pdf or https://arxiv.org/pdf/1803.09047.pdf)
