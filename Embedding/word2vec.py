
from six.moves import urllib
import tensorflow as tf
import os
from tempfile import gettempdir
import zipfile
import collections

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    print(local_filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
        
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified ", filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + 
                        '. Can you get to it with a browser?')
        
    return local_filename

filename = maybe_download("text8.zip", 31344016)

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        for name in f.namelist():
            print(name)
         
        print(type(f.read(f.namelist()[0])))   
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    
    return data

vocabolary = read_data(filename)
print("Data size: ", len(vocabolary))

vocabolary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word,0)
        if index == 0:
            unk_count += 1
        data.append(index)
    
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


if __name__ == '__main__':
    filename = maybe_download("text8.zip", 31344016)
    read_data(filename)