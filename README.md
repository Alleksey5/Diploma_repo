# \# HiFi++ and Deep Filtering for Speech Bandwidth Extension

# 

# \## Overview

# 

# This repository contains the implementation and experimental study of neural architectures for \*\*Speech Bandwidth Extension (BWE)\*\* based on the \*\*HiFi++\*\* framework.

# 

# The primary goal of the project is to investigate the integration of \*\*Deep Filtering\*\* techniques into generative audio restoration models and to analyze the trade-off between reconstruction quality and computational complexity.

# 

# The work was conducted as part of a Bachelor's Thesis focused on restoring wideband speech from narrowband recordings.

# 

# \---

# 

# \## Task

# 

# The considered Speech Bandwidth Extension task consists of reconstructing missing high-frequency components of speech.

# 

# In all experiments the model restores speech sampled at:

# 

# \* Input: \*\*4 kHz\*\*

# \* Output: \*\*16 kHz\*\*

# 

# The objective is to recover high-frequency information while preserving speech naturalness and intelligibility.

# 

# \---

# 

# \## Baseline Architecture

# 

# The baseline model is based on the HiFi++ architecture and consists of the following modules:

# 

# \* \*\*SpectralUNet\*\* — spectral feature restoration in the mel-spectrogram domain;

# \* \*\*HiFi-GAN Upsampler\*\* — conversion of spectral features into waveform representations;

# \* \*\*WaveUNet\*\* — waveform refinement;

# \* \*\*SpectralMaskNet\*\* — spectral post-processing module.

# 

# \---

# 

# \## Deep Filtering Experiments

# 

# Several modifications of the baseline architecture were implemented and evaluated.

# 

# \### Experiment 1: SpectralMaskNet → Deep Filtering

# 

# The SpectralMaskNet module is completely replaced with a Deep Filtering block operating in the time-frequency domain.

# 

# \### Experiment 2: Feature Deep Filtering

# 

# Deep Filtering is applied directly in the latent feature space of the generator before waveform reconstruction.

# 

# \### Experiment 3: DF Encoder Add

# 

# The DeepFilterNet encoder is used as an additional feature extraction branch whose embeddings are fused with HiFi++ representations.

# 

# \### Experiment 4: Conditioned Deep Filtering

# 

# Deep Filtering coefficients are conditioned on hidden features produced by the HiFi++ generator.

# 

# \---

# 

# \## Datasets

# 

# \### VCTK

# 

# \* 109 English speakers

# \* High-quality speech recordings

# \* Resampled to 16 kHz

# 

# The dataset is used for training and evaluation of all models.

# 

# \---

# 

# \## Evaluation Metrics

# 

# The following objective and perceptual metrics are used:

# 

# \### Spectral Metrics

# 

# \* LSD

# \* LSD-LF

# \* LSD-HF

# 

# \### Speech Quality Metrics

# 

# \* PESQ

# \* STOI

# \* COVL

# \* CSIG

# \* CBAK

# 

# \---

# 

# \## Main Findings

# 

# The experiments show that:

# 

# \* Deep Filtering improves spectral reconstruction quality, especially in the high-frequency region.

# \* Direct replacement of SpectralMaskNet achieves the best spectral metrics.

# \* Full DeepFilterNet integration substantially increases model size.

# \* Lightweight integration strategies preserve most of the quality improvements while significantly reducing the number of parameters.

# 

# \---

# 

# \## Technologies

# 

# \* PyTorch

# \* Hydra / OmegaConf

# \* Comet ML

# \* Torchaudio

# \* Librosa

# 

# \---

# 

# \## Repository Structure

# 

# ```text

# src/

# ├── model/

# │   ├── hifigan.py

# │   ├── deep\_filter\_net.py

# │   └── ...

# ├── trainer/

# ├── datasets/

# ├── metrics/

# ├── loss/

# └── configs/

# ```

# 

# \---

# 

# \## References

# 

# \* HiFi++: A Unified Framework for Bandwidth Extension and Speech Enhancement

# \* DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement

# \* AERO: Audio Super-Resolution in the Spectral Domain

# \* DDSP: Differentiable Digital Signal Processing

# \* VM-ASR: Lightweight Audio Super-Resolution with State Space Models



