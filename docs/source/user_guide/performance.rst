

.. _performance:

***************************
Troubleshooting performance
***************************

There are a number of optimizations that you can do to your corpus to speed up MFA or make it more accurate.

Speed optimizations
===================

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
                          command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-nostats', '-i', path '-acodec' 'pcm_s16le' '-f' 'wav', '-ar', '16000', resampled_file]
                      else:
                          command = ['sox', path, '-t', 'wav' '-r', '16000', '-b', '16', resampled_file]
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
