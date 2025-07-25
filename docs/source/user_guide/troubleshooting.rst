

.. _troubleshooting:

***************
Troubleshooting
***************

General Tips
============

* Use dedicated Conda environments
  * Don't be afraid to recreate environments
  * Particularly if you get environment solver errors, it's generally easier to redo the commands in :ref:`installation` to get to a working environment than fixing dependency hell
  * Software is disposable and easily reinstalled, data is not
* If something isn't working, first run :code:`mfa_update` to update to the latest available versions
  * If things still aren't working, try running commands with the :code:`--clean` flag to ensure no temporary files are interfering with the run
* Feel free to delete the MFA temporary directory as necessary (defaults to :code:`~/Documents/MFA`), it is made to be temporary
  * Even if you're using PostGreSQL databases, they are geared towards shorter term use
  * If you've run :ref:`configure_cli` in the past, you'll need to rerun it, as the configuration is saved in the temporary directory
* Please use the search functionality of the docs to see if your issue or question is answered in the docs, they are generally quite extensive
  * If you can't find the answer to your question, please feel free to email michael.e.mcauliffe@gmail.com and it'll inspire me to add more to the relevant documentation


Errors aligning single files
============================

In general, MFA is not intended to align single files, particularly if they are long, have noise in the background, a different style such a singing etc.  Some aspects of these can be improved by aligning a larger set of utterances per speaker, as successful alignments on the first pass can aid second pass alignments through feature-space transforms.

If you must align a single file and run into an error, you can increase the beam width via :code:`--beam 100`, this will result in less optimal alignments being generated.  The beam width is intentionally set low so that the fMLLR feature space transform for speaker adaptation is using quality alignments and not potential noise or bad alignments.

.. seealso::

   See :ref:`align_one` for a command geared towards aligning a single file rather than :code:`mfa align`.

Errors training on small corpora
================================

MFA generally needs a large set of utterances and speakers to train models.  I typically shoot for at least 1000 hours of data for the pretrained models for MFA.  If you do not need a model to be generalizable (i.e. you're just using it on the data that it was trained on to generate alignments for a small corpus), then you do not need as much data, but you will need at least several hours worth. In general, the more the better and the more variation you can include in the form of speakers and utterances per speaker the better.  Obviously training data quality should be inspected, as models and overall alignments can be negatively impacted by noisy files (i.e., files without speech, low SNR, clipping and stuttering, etc).


Different numbers of speakers
=============================

Please refer to :ref:`corpus_structure` for how your corpus directory should be structured for MFA to correctly parse speaker information.

Improving alignment quality
===========================

Add pronunciations to the pronunciation dictionary
--------------------------------------------------

Pretrained models are trained a particular dialect/style, and so adding pronunciations more representative of the variety spoken in your dataset will help alignment.

Check the quality of your data
------------------------------

Double check that your transcription files match the audio.  Ensure that hesitation words like "uh" and "um" are represented, as well as cutoffs or hesitations.

* See :ref:`validating_data` for more information on running MFA's validate command, which aims to detect issues in the dataset.
* Use MFA's `anchor utility <https://anchor-annotator.readthedocs.io/en/latest/>`_ to visually inspect your data as MFA sees it and correct issues in transcription or OOV items.

Adapt the model to your data
----------------------------

See :ref:`adapt_acoustic_model` for how to adapt some of the model parameters to your data based on an initial alignment, and then run another alignment with the adapted model.

Speed optimizations
===================

There are a number of optimizations that you can do to your corpus to speed up MFA or make it more accurate.

.. _wav_conversion:

Convert to basic wav files
--------------------------

In general, 16kHz, 16-bit wav files are lingua franca of audio, though they are uncompressed and can take up a lot of space.  However, if space is less an issue than processing time, converting all your files to wav format before running MFA will result in faster processing times (both load and feature generation).

Script example
``````````````

.. warning::

   This script modifies audio files in place and deletes the original file.  Please back up your data before running it if you only have one copy.

.. code-block:: python

   import os
   import subprocess
   import sys

   corpus_directory = '/path/to/corpus'

   file_extensions = ['.flac', '.mp3', '.wav', '.aiff']

   def wavify_sound_files():
       for speaker in os.listdir(corpus_directory):
           speaker_dir = os.path.join(corpus_directory, speaker)
           if not os.path.isdir(speaker_dir):
               continue
           for file in os.listdir(speaker_dir):
               for ext in file_extensions:
                  if file.endswith(ext):
                      path = os.path.join(speaker_dir, file)
                      if ext == '.wav':
                         resampled_file = path.replace(ext, f'_fixed{ext}')
                      else:
                         resampled_file = path.replace(ext, f'.wav')
                      if sys.platform == 'win32' or ext in {'.opus', '.ogg'}:
                          command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-nostats', '-i', path, '-acodec', 'pcm_s16le', '-f', 'wav', '-ar', '16000', resampled_file]
                      else:
                          command = ['sox', path, '-t', 'wav', '-r', '16000', '-b', '16', resampled_file]
                      subprocess.check_call(command)
                      os.remove(path)
                      os.rename(resampled_file, path)

   if __name__ == '__main__':
       wavify_sound_files()

.. note::

   This script assumes that the corpus is already adheres to MFA's supported :ref:`corpus_structure` (with speaker directories of their files under the corpus root), and that you are in the conda environment for MFA.

Downsample to 16kHz
-------------------

Both Kaldi and SpeechBrain operate on 16kHz as the primary sampling rate.  If your files have a sampling rate greater than 16kHz, then every time they are processed (either as part of MFCC generation in Kaldi, or in running SpeechBrain's VAD/Speaker classification models), there will be extra computation as they are downsampled to 16kHz.

.. note::

   As always, I recommend having an immutable copy of the original corpus that is backed up and archived separate from the copy that is being processed.


Script example
``````````````

.. warning::

   This script modifies the sample rate in place and deletes the original file.  Please back up your data before running it if you only have one copy.

.. code-block:: python

   import os
   import subprocess

   corpus_directory = '/path/to/corpus'

   file_extensions = ['.wav', '.flac']

   def fix_sample_rate():

       for speaker in os.listdir(corpus_directory):
           speaker_dir = os.path.join(corpus_directory, speaker)
           if not os.path.isdir(speaker_dir):
               continue
           for file in os.listdir(speaker_dir):
               for ext in file_extensions:
                  if file.endswith(ext):
                      path = os.path.join(speaker_dir, file)
                      resampled_file = path.replace(ext, f'_resampled{ext}')
                      subprocess.check_call(['sox', path, '-r', '16000', resampled_file])
                      os.remove(path)
                      os.rename(resampled_file, path)

   if __name__ == '__main__':
       fix_sample_rate()

.. note::

   This script assumes that the corpus is already adheres to MFA's supported :ref:`corpus_structure` (with speaker directories of their files under the corpus root), and that you are in the conda environment for MFA.

Change bit depth of wav files to 16bit
--------------------------------------

Kaldi does not support ``.wav`` files that are not 16 bit, so any files that are 24 or 32 bit will be processed by ``sox``.  Changing the bit depth of processed wav files ahead of time will save this computation when MFA processes the corpus.


Script example
``````````````

.. warning::

   This script modifies the bit depth in place and deletes the original file.  Please back up your data before running it if you only have one copy.

.. code-block:: python

   import os
   import subprocess

   corpus_directory = '/path/to/corpus'


   def fix_bit_depth():

       for speaker in os.listdir(corpus_directory):
           speaker_dir = os.path.join(corpus_directory, speaker)
           if not os.path.isdir(speaker_dir):
               continue
           for file in os.listdir(speaker_dir):
               if file.endswith('.wav'):
                   path = os.path.join(speaker_dir, file)
                   resampled_file = path.replace(ext, f'_resampled{ext}')
                   subprocess.check_call(['sox', path, '-b', '16', resampled_file])
                   os.remove(path)
                   os.rename(resampled_file, path)

   if __name__ == '__main__':
       fix_bit_depth()

.. note::

   This script assumes that the corpus is already adheres to MFA's supported :ref:`corpus_structure`, and that you are in the conda environment for MFA.
