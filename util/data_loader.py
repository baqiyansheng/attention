"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.legacy.data import Field, BucketIterator, Example, Dataset
from torchtext.legacy.datasets.translation import Multi30k
import ssl
import urllib3
import json

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token,
                 train_file, valid_file, test_file):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        print('dataset initializing start')

    def load_data(self, file_path, fields):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    item = json.loads(line)
                    # 读取 'en' 和 'de' 字段
                    src = item['en']
                    trg = item['de']
                    examples.append(Example.fromlist([src, trg], fields))
        return examples

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)
            self.target = Field(tokenize=self.tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)
            self.target = Field(tokenize=self.tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                lower=True,
                                batch_first=True)

        fields = [('src', self.source), ('trg', self.target)]

        train_examples = self.load_data(self.train_file, fields)
        valid_examples = self.load_data(self.valid_file, fields)
        test_examples = self.load_data(self.test_file, fields)

        train_data = Dataset(train_examples, fields)
        valid_data = Dataset(valid_examples, fields)
        test_data = Dataset(test_examples, fields)

        return train_data, valid_data, test_data
        # train_data, valid_data, test_data = Multi30k.splits(
        #     exts=self.ext, fields=(self.source, self.target))
        # return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test), batch_size=batch_size, device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
