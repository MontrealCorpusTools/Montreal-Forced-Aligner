import time
import os
import platform
import csv
from datetime import datetime

import sys
sys.path.insert(0,'/Users/michaelasocolof/Prosodylab-Kaldi-Aligner')

from aligner.aligner import BaseAligner

thedict = os.path.expanduser('~/benchmarking/librispeech-lexicon.txt')
corpus = os.path.expanduser('smallLibriSpeech')
generated_dir = 'generated_dir'

def aligner_test(thedict, corpus, generated_dir):
    a = BaseAligner(sick_corpus, sick_dict, os.path.join(generated_dir,'sick_output'),
                        temp_directory = os.path.join(generated_dir,'sickcorpus'))
    a.train_mono()

librispeech_corpus = corpus_basic(thedict, corpus, generated_dir)

def WriteDictToCSV(csv_file,csv_columns,dict_data):
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)   
        return

csv_columns = ['Computer','Date','Corpus', 'Type of benchmark', 'Total time']
dict_data = [
    {'Computer': platform.node(), 'Date': str(datetime.now()), 'Corpus': 'librispeech', 'Type of benchmark': 'aligner', 'Total time': librispeech_corpus[0]}]

now = datetime.now()
date = str(now.year)+str(now.month)+str(now.day)

if not os.path.exists('alignerbenchmark'+date+'.csv'):
    open('alignerbenchmark'+date+'.csv', 'a')

csv_file = 'alignerbenchmark'+date+'.csv'

with open('alignerbenchmark'+date+'.csv', 'a') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writerow(dict_data[0])