from scipy import spatial
from scipy import stats
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn import metrics


def get_file_names(path, filter):
    """returns a list of all files' names in the given directory and its sub-folders"""
    if os.path.isfile(path):
        file_list = [path]
    else:
        file_list = []
        for path, subdirs, files in os.walk(path):
            for name in files:
                file_list.append(os.path.join(path, name))
        file_list.sort()
        file_list = [file for file in file_list if filter in file]
    return file_list

def seq_check(path, start_ref='last', plot=True, filter=''):
    files_list = get_file_names(path, filter=filter)
    check = {}
    if start_ref == 'last':
        ref = tiff.imread(files_list[-1])
        scope = np.arange(len(files_list)-1, -1, -1)
    else:
        ref = tiff.imread(files_list[0])
        scope = np.arange(0,len(files_list)-1,1)
    for ind in scope:
        file = files_list[ind]
        image = tiff.imread(file)
        value = sum(metrics.pairwise.cosine_similarity(image.ravel().reshape(1,-1), 
                    ref.ravel().reshape(1,-1)))
        check[ind] = value
        ref = image
        print('finished checking ', os.path.basename(file), ind, value)
    if plot == True:
        check_list = check.items()
        x, y = zip(*check_list)
        print(x)
        print(y)
        plt.plot(x, y)
        plt.xlabel('stack #')
        plt.ylabel('distance value')
        plt.xticks(np.arange(0, len(x)+5, step=5))
        plt.savefig(path+'check.pdf')
        print('plot is created')
    return check


def fixed_check(path, ref_label='', plot=True, filter=''):
    files_list = get_file_names(path, filter=filter)
    files_list = [file for file in files_list if '.tif' in file]
    ref_file = [file for file in files_list if ref_label in file][0]
    ref = tiff.imread(ref_file)
    # check = {os.path.basename(ref_file):(1 - spatial.distance.cosine(ref.ravel(), ref.ravel()))}
    # print(check)
    # files_list = [file for file in files_list if ref_label not in file]
    check = {}
    # print(len(files_list))
    for file in files_list:
        image = tiff.imread(file)
        key_label = os.path.basename(file).partition('_')[0]
        print(key_label)
        check[key_label] = sum(metrics.pairwise.cosine_similarity(image.ravel().reshape(1,-1), 
                                ref.ravel().reshape(1,-1)))
        print(check[key_label])
        print('finished checking ', os.path.basename(file))
    if plot == True:
        check_list = check.items()
        x, y = zip(*check_list)
        plt.plot(x, y)
        plt.xlabel('drift method')
        plt.ylabel('distance value')
        plt.xticks(np.arange(0, len(x)+5, step=5))
        plt.savefig(path+'check.pdf')
        print('plot is created')
    return check

def drift_check(path, method='fixed', plot=True, filter='', ref_label='', start_ref='last'):
    if method == 'seq':
        check = seq_check(path, filter=filter, plot=plot, start_ref=start_ref)
    elif method == 'fixed':
        check = fixed_check(path, ref_label=ref_label, plot=plot, filter=filter)
    with open(path+'check.csv', 'w') as f:
        for key in check.keys():
            f.write("%s,%s\n"%(key,check[key][0]))
    return check

def main():
    parser = argparse.ArgumentParser(description='apply scipy.distance to check registration performance',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='provide path to your drift corrected data')
    parser.add_argument('-m','--method', help='provide path to your drift corrected data')
    parser.add_argument('-r','--ref', help='provide path to your drift corrected data')
    parser.add_argument('-f','--filter', help='provide path to your drift corrected data')
    parser.add_argument('-s','--start', help='start reference with first or last?')
    args = parser.parse_args()
    check = drift_check(args.path, method=args.method, filter=args.filter, plot=True, ref_label=args.ref, start_ref=args.start)
    return check


if __name__ == '__main__':
    main()