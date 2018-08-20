# Simple script to pull a graph out for accuracy vs training iteration
# for nnet2 implementation in MFA.
# Probably should be integrated with MFA for # jobs parameters, etc.
# eventually.

import sys
import os
import glob
#import numpy as np
#from matplotlib import pyplot as plt

def get_accuracy_graph(log_dir, export_dir):
    os.chdir(log_dir)
    compute_prob_logs = glob.glob('compute_prob_train.*.log')
    acc_pairs = []
    prob_pairs = []
    iterations = []
    for log in compute_prob_logs:
        print(log)
        split_name = log.split('.')
        iteration = split_name[1]
        if iteration == 'final':
            iteration = int(max(iterations))+1
        iterations.append(int(iteration))
        with open(log, 'r') as fp:
            lines = fp.readlines()
            #print(lines)
            index = len(lines)-1
            line = lines[index]
            #while not line.startswith('LOG (nnet-compute-prob:main()'):
            while not 'and accuracy is' in line:
                #print(line)
                #print(index)
                index = index-1
                line = lines[index]
            accuracy = line.split(' ')[12]
            prob = line.split(' ')[8]
            acc_pair = [int(iteration), float(accuracy)]
            prob_pair = [int(iteration), float(prob)]
            print(prob_pair)
            acc_pairs.append(acc_pair)
            prob_pairs.append(prob_pair)
            acc_pairs.sort(key=lambda x: x[0])
            prob_pairs.sort(key=lambda x: x[0])

    os.chdir(export_dir)

    plt.gcf().clear()
    acc_pairs = np.array(acc_pairs)
    x, y = acc_pairs.T
    plt.scatter(x, y)
    plt.title("# Iterations vs. Accuracy")
    plt.xlabel("# iterations")
    plt.ylabel("accuracy")
    plt.savefig('accuracy.png')

    plt.gcf().clear()
    prob_pairs = np.array(prob_pairs)
    x, y = prob_pairs.T
    plt.scatter(x, y)
    plt.title("# Iterations vs. Log Prob")
    plt.xlabel("# iterations")
    plt.ylabel("log_prob")
    plt.savefig('log_prob.png')

if __name__ == '__main__':
    log_dir = sys.argv[1]
    get_accuracy_graph(log_dir, log_dir)
