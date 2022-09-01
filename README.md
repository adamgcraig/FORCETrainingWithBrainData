# FORCETrainingWithBrainData
Applying and building on the code from https://github.com/ModelDBRepository/190565 with the goal of creating an interpretable model that can learn to mimic individual whole-brain dynamics.

The model comes from this paper:
Nicola, W., & Clopath, C. (2017).
Supervised learning in spiking neural networks with FORCE training.
Nature communications, 8(1), 1-15.

I started with Nicola et al.'s script and made minor changes step-by-step in order to get it to work with the fMRI input data, make it easier to use, and generate the figures I wanted.
To avoid legal or patient privacy issues with redistributing Human Connectome Project data, this publicly available repository has a script to generate sham data using a network of coupled oscillators.