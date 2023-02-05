
(fine_tune_alignments)=

# Fine-tuning alignments

By default and standard in ASR, the frame step between feature frames is set to 10 ms, which limits the accuracy of MFA to a minimum of 0.01 seconds. When the `--fine_tune` flag is specified, the aligner does an extra fine-tuning step following alignment. The audio surrounding each interval's initial boundary is extracted with a frame step of 1 ms (0.001s) and is aligned using a simple phone dictionary combined with a transcript of the previous phone and the current phone.  Extracting the phone alignment gives the possibility of higher degrees of accuracy (down to 1ms).

:::{warning}

The actual accuracy bound is not clear as each frame uses the surrounding 25ms to generate features, so each frame necessary incorporates time-smeared acoustic information.
:::
