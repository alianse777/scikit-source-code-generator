# Copyright 2017 alianse777
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np
import pickle as pkl
import sys
import os
import argparse

class Generator:
    def __init__(self, mlen=128):
        self.mlen = mlen # size of windows
        self.step = 1
        self.vect = OneHotEncoder()
        open("input", "wb").close()
        open("output", "wb").close()

    def load_net(self):
        with open("index.pkl", "rb") as fl:
            self.char_index, self.index_char, self.vect = pkl.load(fl)
        self.model = joblib.load("model.x")

    def build_model(self, shape):
        self.model = DecisionTreeRegressor()

    def train(self, datafile, read_batch):
        with open(datafile, "r") as fl:
            self.txt = fl.read(read_batch).lower().replace("    ", " ").replace("\t", " ")
        self.txt = re.sub(r'[^\x00-\x7F]+','', self.txt) # remove non-ASCII chars
        self.chars = list(set(self.txt))
        print ("Text length:", len(self.txt))
        print ("Chars length:", len(self.chars))
        en = list(enumerate(self.chars))
        self.char_index = {x[1]: x[0] for x in en}
        self.index_char = {x[0]: x[1] for x in en}
        self.sentences = (self.txt[i-self.mlen:i] for i in range(self.mlen, len(self.txt), self.step))
        self.outputs = (self.txt[i] for i in range(self.mlen, len(self.txt), self.step))
        self.size = len(self.txt) - self.mlen
        print ("Vectorization...")
        X, y = self.encode(self.sentences, self.outputs)
        self.build_model((self.mlen, len(self.chars)))
        X = self.vect.fit_transform(X)
        # dump all necessary objects
        pkl.dump((self.char_index, self.index_char, self.vect), open("index.pkl", "wb"))
        print ("Training...")
        self.model.fit(X, y)
        joblib.dump(self.model, "model.x")
        
    def generate(self, seed, length=1):
        print ("Generating...")
        seed = re.sub(r'[^\x00-\x7F]+','', seed).replace("    ", " ").replace("\t", "    ")
        generated = seed[:self.mlen].lower()
        for i in range(length):
            tmp = self.encode_raw([generated[-self.mlen:]])
            data = self.decode_flat(self.model.predict(self.vect.transform(tmp)))
            generated += data
        return generated[self.mlen:]
        
    def encode(self, sentences, outputs):
        m = "readwrite"
        X = np.memmap("input", shape=(self.size, self.mlen), dtype="int8", mode=m)
        y = np.memmap("output", shape=(self.size, len(self.chars)), dtype="bool", mode=m)
        for (i, sentence), out_char in zip(enumerate(sentences), outputs):
            for v, char in enumerate(sentence):
                X[i, v] = self.char_index[char]
            y[i, self.char_index[out_char]] = 1
        return X, y
        
        
    def encode_raw(self, sentences):
        X = np.zeros((len(sentences), self.mlen))
        for i, sentence in enumerate(sentences):
            for v, char in enumerate(sentence):
                X[i, v] = self.char_index[char]
        return X
        
    def decode_flat(self, matrix):
        result = ""
        for i in matrix:
            lt = i.tolist()
            result += self.index_char[lt.index(max(lt))]
        return result
    
    def decode_x(self, matrix):
        result = []
        for i in matrix:
            result.append(self.decode_flat(i))
        return result

if __name__ == "__main__":
    generator = Generator()
    parser = argparse.ArgumentParser(description='Python code generator')
    parser.add_argument('--train', dest='n_train', type=int, 
        help='Train regression model on file with code on N characters')
    parser.add_argument('--generate', dest='n_generate', type=int,
        help='Generate N chars python source code from given seed file. ')
    parser.add_argument('file', metavar='FILE', type=str, 
        help='File with sample code or seed for generation.')
    args = parser.parse_args()
    if args.n_train:
        if not os.path.isfile(args.file):
            print ("File not found!")
            sys.exit(-1)
        generator.train(args.file, args.n_train)
    elif args.n_generate:
        if not os.path.isfile(args.file):
            print ("File not found!")
            sys.exit(-1)
        with open(args.file, "r") as fl:
            generator.load_net()
            print (generator.generate(fl.read(args.n_generate), args.n_generate))
    else:
        print ("Type -h for help.")
        
