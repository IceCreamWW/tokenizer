import pygtrie as trie
import re
import math
import pickle
import os
from functools import reduce


class Word:
    def __init__(self, text, frequency=0):
        self.text = text
        self.length = len(self.text)
        self.frequency = frequency

    def __str__(self):
        return self.text


class Chunk:
    def __init__(self, w1, w2=None, w3=None):
        self.words = [w1]
        if w2 is not None:
            self.words.append(w2)
        if w3 is not None:
            self.words.append(w3)

    @property
    def total_length(self):
        return reduce(lambda x, y: x + y.length, self.words, 0)

    @property
    def average_length(self):
        return float(self.total_length) / float(len(self.words))

    @property
    def standard_deviation(self):
        return math.sqrt(reduce(lambda x, y: x + (y.length - self.average_length)**2, self.words, 0.0) / self.total_length)

    @property
    def word_frequency(self):
        return reduce(lambda x, y: x + y.frequency if y.length == 1 else x, self.words, 0)


class Tokenizer:
    digits_pattern = r'￥?[0-9０１２３４５６７８９０.]*[0-9０１２３４５６７８９０]'
    digits_placeholder = '灅'

    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_placeholder = '燚'
    punctuations = set()

    def __init__(self, train_source='./data/train.txt', use_exist=True):
        self.vocabulary = trie.CharTrie()
        self.all_punctuations = None
        self.max_word_frequency = 0
        self.__build_punctuations(path_source='./data/punctuations.txt')
        self.__build_vocabulary(path_source=train_source, use_exist=use_exist)

        if len(Tokenizer.punctuations) == 0:
            Tokenizer.punctuations = set(self.all_punctuations)

    def cut(self, sentence, delimiter='\t'):
        url_backup = re.findall(Tokenizer.url_pattern, sentence)
        sentence = re.sub(Tokenizer.url_pattern, Tokenizer.url_placeholder, sentence)

        digits_backup = re.findall(Tokenizer.digits_pattern, sentence)
        sentence = re.sub(Tokenizer.digits_pattern, Tokenizer.digits_placeholder, sentence)

        cursor = 0
        words = []

        while cursor < len(sentence):
            if Tokenizer.is_punctuation(sentence[cursor]):
                cur_punc = sentence[cursor]
                words.append(sentence[cursor])
                cursor += 1
                if cur_punc in '.。':
                    while cursor < len(sentence) and sentence[cursor] == cur_punc:
                        words[-1] += sentence[cursor]
                        cursor += 1
            elif Tokenizer.is_english_char(sentence[cursor]):
                word = self.match_word(sentence[cursor:], match_type='en')
                words.append(word.text)
                cursor += word.length
            else:
                chunks = self.get_chunks(sentence[cursor:])
                if len(chunks) > 0:
                    words_ex = self.resolve_ambiguity(chunks)
                    words.extend([word.text for word in words_ex])
                    cursor += sum([word.length for word in words_ex])
                else:
                    words.append(sentence[cursor])
                    cursor += 1

        # 还原数字
        result = delimiter.join([word for word in words if len(word) != 0])
        for backup in digits_backup:
            result = result.replace(Tokenizer.digits_placeholder, backup, 1)

        # 还原url
        for backup in url_backup:
            result = result.replace(Tokenizer.url_placeholder, backup, 1)

        return result

    def resolve_ambiguity(self, chunks):
        # 根据总词长消歧
        if len(chunks) > 1:
            score = max([x.total_length for x in chunks])
            chunks = [chunk for chunk in chunks if chunk.total_length == score]

        # 根据总词长消歧
        if len(chunks) > 1:
            score = max([x.average_length for x in chunks])
            chunks = [chunk for chunk in chunks if chunk.average_length == score]

        # 根据平均词长消歧
        if len(chunks) > 1:
            score = max([x.standard_deviation for x in chunks])
            chunks = [chunk for chunk in chunks if chunk.standard_deviation == score]

        # 根据最大单字词频消歧
        if len(chunks) > 1:
            score = max([x.word_frequency for x in chunks])
            chunks = [chunk for chunk in chunks if chunk.word_frequency == score]

        if len(chunks) != 1:
            print("无法消除歧义, 默认选择最后一个")
            return chunks[-1].words

        return chunks[0].words

    def get_chunks(self, sentence):
        chunks = []

        chunk_begin = self.match_word(sentence)

        for b in chunk_begin:
            if b.length > 0:
                chunk_middle = self.match_word(sentence[b.length:])
                if len(chunk_middle) == 0:
                    chunks.append(Chunk(b))
                for m in chunk_middle:
                    if m.length > 0:
                        chunk_end = self.match_word(sentence[b.length + m.length:])
                        if len(chunk_end) == 0:
                            chunks.append(Chunk(b, m))
                        for e in chunk_end:
                            if e.length > 0:
                                chunks.append(Chunk(b, m, e))
                            else:
                                chunks.append(Chunk(b, m))
                    else:
                        chunks.append(Chunk(b))
        return chunks

    def match_word(self, sentence, match_type='ch'):
        # 中文单词
        if match_type == 'ch':
            return [item.value for item in self.vocabulary.prefixes(sentence)]
        elif match_type == 'en':
            cursor = 0
            while cursor < len(sentence) and Tokenizer.is_english_char(sentence[cursor]):
                cursor += 1
            return Word(sentence[:cursor])
        else:
            return Word(sentence[0])

    @staticmethod
    def is_english_char(character):
        return (character >= 'a' and character <= 'z') or (character >= 'A' and character <= 'Z')

    @staticmethod
    def is_punctuation(character):
        return character in Tokenizer.punctuations

    def __build_punctuations(self, path_source):
        with open(path_source, encoding='u8') as fp:
            self.all_punctuations = set(fp.read().split('\n'))

    def __build_vocabulary(self, path_source, use_exist):
        if use_exist and os.path.isfile('./model/vocabulary.pkl'):
            with open('./model/vocabulary.pkl', 'rb') as fp:
                self.vocabulary, self.max_word_frequency = pickle.load(fp)

        else:
            if path_source.endswith('.dic') or path_source.endswith('.dict'):
                self.__build_vocabulary_from_dict(path_source)
            else:
                self.__build_vocabulary_from_text(path_source)

            dump_path = './model/vocabulary.pkl' if use_exist else './model/vocabulary-custom.pkl'
            with open(dump_path, 'wb') as fp:
                pickle.dump((self.vocabulary, self.max_word_frequency), fp)

    def __build_vocabulary_from_dict(self, path_dict, delimiter='\t'):
        with open(path_dict, encoding='u8') as fp:
            records = fp.read().split('\n')
            for record in records:
                word, frequency = record.split(delimiter)
                self.vocabulary[word] = Word(word, frequency)
                self.max_word_frequency = max(self.max_word_frequency, frequency)

    def __build_vocabulary_from_text(self, path_text, delimiter='\t'):
        with open(path_text, encoding='u8') as fp:
            corpus = re.sub(Tokenizer.digits_pattern, Tokenizer.digits_placeholder, fp.read())
            corpus = corpus.split('\n')
            for line in corpus:
                for word in line.split(delimiter):
                    if word in self.all_punctuations:
                        if word not in self.punctuations:
                            self.punctuations.add(word)
                            break

                    if word not in self.vocabulary:
                        self.vocabulary[word] = Word(word)
                    else:
                        self.vocabulary[word].frequency += 1

                    self.max_word_frequency = max(self.max_word_frequency, self.vocabulary[word].frequency)

        # 赋予含量词的数字比纯数字更大的权重,用于消歧
        if Tokenizer.digits_placeholder in self.vocabulary:
            for word in self.vocabulary.iteritems(prefix=Tokenizer.digits_placeholder):
                word[1].frequency = self.max_word_frequency * 2
            self.vocabulary[Tokenizer.digits_placeholder].frequency = self.max_word_frequency


def test_sentence(sentence):
    tk = Tokenizer()
    print(tk.cut(sentence, delimiter='/'))


def test_all(corpus):
    tk = Tokenizer()
    inputfile = ''
    outputfile = ''
    if corpus == 'dev1':
        inputfile = './test/judge.data.1'
        outputfile = './test/result.mmseg.1.txt'
    elif corpus == 'dev2':
        inputfile = './test/judge.data.2'
        outputfile = './test/result.mmseg.2.txt'
    elif corpus == 'test1':
        inputfile = './test/final-test.1.txt'
        outputfile = './test/final-test.result.mmseg.1.txt'
    elif corpus == 'test2':
        inputfile = './test/final-test.2.txt'
        outputfile = './test/final-test.result.mmseg.2.txt'

    with open(inputfile, 'r', encoding='u8') as fin:
        with open(outputfile, 'w', encoding='u8') as fout:
            for line in fin.read().split('\n'):
                fout.write(tk.cut(line) + '\n')


if __name__ == "__main__":
    test_sentence('我国律师工作是随着改革开放和民主法制建设的加强而建立和发展起来的。')
    # test_all('test2')
