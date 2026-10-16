# TAP-Syn
The source code of paper "Target-Adaptive Proxy Domain Synthesis for DL-FWI".

Supervised deep learning-based full waveform inversion (DL-FWI) learns a mapping from seismic records to velocity models, but is typically hindered by the scarcity of labeled target samples.
However, directly fine-tuning a source-pretrained inversion network on limited target samples may provide insufficient supervision for target-domain inversion.
In this paper, we propose Target-Adaptive Proxy Domain Synthesis (TAP-Syn) to construct a target-adaptive proxy domain by combining original target samples with forward-modeled real-anchor and generative expansions, using coarse-conditioned residual rectified flow and coupled LoRA adaptation.

Train code files: train.py(main running program)

The primary procedure for training our method. The model checkpoints generated during training will be stored in the output directory specified by --output.

Test code files: test.py (main running program)

The main program for testing the model. The evaluation metric results generated during the test will be printed to the terminal.
