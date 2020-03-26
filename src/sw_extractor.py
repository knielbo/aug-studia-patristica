#!/home/knielbo/virtenvs/teki/bin/python
"""
Corpus-specific stopword extractor

Usage:
    python sw_extractor.py --dataset <path> --length <int>

    dataset: filepath to directory contaning files
    length: length of list
"""
import os
import glob
import re
from natsort import natsorted
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from nltk.tokenize.punkt import PunktLanguageVars
from cltk.stop.latin import CorpusStoplist
import argparse

class Corpus:
    def __init__(self, filepath):
        self.path = filepath

    def read(self, fileformat=".txt"):
            fnames = natsorted(glob.glob(os.path.join(self.path, "*" + fileformat)))
            texts = list()
            for fname in fnames:
                with open(fname, "r") as f:
                    text = f.read()
                
                texts.append(text)
            
            fnames = [os.path.basename(fname).split(".")[0] for fname in fnames]
            
            return texts, fnames


class LemmatizerLatin:
    def __init__(self, token=True):
        self.lemmatizer = BackoffLatinLemmatizer()
        self.token = token
    
    def preprocess(self, text):
        if self.token:
            lemma = self.lemmatizer.lemmatize(text)
        else:
            plv = PunktLanguageVars()
            unigrams = plv.word_tokenize(text)
            lemma = self.lemmatizer.lemmatize(unigrams)
        
        lemma = [t[0] if t[1] == "punc" else t[1] for t in lemma]    
        
        return " ".join(lemma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-l", "--length", required=True, help="length of sw list")
    
    args = vars(ap.parse_args())
    print("[INFO] Importing data...")
    filepath = args["dataset"]  
    C = Corpus(filepath)
    data, _ = C.read()
    data = [file.lower() for file in data]
    pat0 = re.compile(r"\W+",flags=re.MULTILINE)
    data = [pat0.sub(" ", file) for file in data]
    pat1 = re.compile(r"\d+",flags=re.MULTILINE)
    data = [pat1.sub(" ", file) for file in data]
    pat2 = re.compile(r"  +",flags=re.MULTILINE)
    data = [pat2.sub(" ", file) for file in data]
    
    print("[INFO] Lemmatization...")
    le = LemmatizerLatin(token=False)
    data = [le.preprocess(file) for file in data]
    S = CorpusStoplist()
    sw_list = S.build_stoplist(data, size=int(args["length"]))
    
    print("[INFO] Writing list to file...")
    if os.path.isdir("../res"):
        pass
    else:
        os.mkdir("../res")
    with open("../res/stopwords.txt", "w") as f:
        for word in sw_list:
            f.write("%s\n" % word)


if __name__=="__main__":
    main()