import os
import re
import json
from functools import reduce


class Tokenizer:

    OUT_OF_OBS = '_OOO_'
    digits_pattern = r'[0-9０１２３４５６７８９０.]+'
    digits_placeholder = '灅'

    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_placeholder = '燚'

    def __init__(self, train_path='./data/train.txt', model_path='./model/hmm-trained.json', use_exist=True):

        self.states = None          # 状态值集合
        self.pi = None              # 初始状态分布
        self.observations = None    # 观察值集合
        self.A = None               # 转移概率矩阵
        self.B = None               # 发射概率矩阵

        self.__build_model(train_path=train_path, model_path=model_path, use_exist=use_exist)

    def cut(self, sentence, delimiter='\t'):
        if len(sentence) == 0:
            return ''
        url_backup = re.findall(Tokenizer.url_pattern, sentence)
        sentence = re.sub(Tokenizer.url_pattern, Tokenizer.url_placeholder, sentence)

        digits_backup = re.findall(Tokenizer.digits_pattern, sentence)
        sentence = re.sub(Tokenizer.digits_pattern, Tokenizer.digits_placeholder, sentence)

        obs = []
        for ch in sentence:
            if ch in self.observations:
                obs.append(ch)
            else:
                obs.append(Tokenizer.OUT_OF_OBS)

        prob, path = self.viterbi(obs)
        words = Tokenizer.decode(path, sentence)

        result = delimiter.join(words)
        for backup in digits_backup:
            result = result.replace(Tokenizer.digits_placeholder, backup, 1)

        # 还原url
        for backup in url_backup:
            result = result.replace(Tokenizer.url_placeholder, backup, 1)

        return result

    def __build_model(self, train_path, model_path, use_exist):

        if use_exist and os.path.isfile(model_path):
            with open(model_path, 'r', encoding='u8') as fp:
                model_param = json.load(fp)
                self.states = model_param['states']
                self.observations = model_param['observations']
                self.pi = model_param['pi']
                self.A = model_param['A']
                self.B = model_param['B']

        else:
            self.train_hmm(corpus_path=train_path)
            dump_path = model_path if use_exist else './model/hmm-custom.json'

            with open(dump_path, 'w', encoding='u8') as fp:
                json.dump(dict({
                    "states": self.states,
                    "observations": self.observations,
                    "pi": self.pi,
                    "A": self.A,
                    "B": self.B
                }), fp=fp, indent=4, ensure_ascii=False)

    def viterbi(self, sentence):
        probs = [{}]
        path = {}

        for state in self.states:
            probs[0][state] = self.pi[state] * self.B[state][sentence[0]]
            path[state] = [state]

        for i in range(1, len(sentence)):
            probs.append({})
            new_path = {}

            for state in self.states:
                (prob, argmax_state) = max([(probs[i-1][prev_state] * self.A[prev_state][state] * self.B[state][sentence[i]], prev_state) for prev_state in self.states])
                probs[i][state] = prob
                new_path[state] = path[argmax_state] + [state]

            path = new_path

        (prob, argmax_state) = max([(probs[len(sentence) - 1][state], state) for state in self.states])
        return prob, path[argmax_state]

    @staticmethod
    def encode_reduce(result, word):
        if len(word) == 1:
            result.append('S')
            return result

        else:
            result.append('B')
            for cursor in range(1, len(word) - 1):
                result.append('M')
            result.append('E')
            return result

    @staticmethod
    def decode(labels, sentence):
        assert len(labels) == len(sentence)

        words = []
        word = ''
        for label, ch in zip(labels, sentence):
            if label == 'S':
                words.append(ch)
            elif label == 'B':
                word = ch
            elif label == 'M':
                word += ch
            else:
                word += ch
                words.append(word)

        if labels[-1] not in 'SE':
            print("分词结果没有以S或E结尾")
            words.append(word)

        return words

    def train_hmm(self, corpus_path, delimiter='\t'):
        self.states = ['B', 'M', 'E', 'S']
        self.observations = [Tokenizer.OUT_OF_OBS]
        self.pi = {'B': .0, 'M': .0, 'E': .0, 'S': .0}

        # transition probability matrix
        self.A = {
            'B': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
            'E': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
            'M': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
            'S': {'B': 0, 'E': 0, 'M': 0, 'S': 0}
        }

        # emission probability matrix
        self.B = {
            'B': {Tokenizer.OUT_OF_OBS: 1},
            'E': {Tokenizer.OUT_OF_OBS: 1},
            'M': {Tokenizer.OUT_OF_OBS: 1},
            'S': {Tokenizer.OUT_OF_OBS: 1}
        }

        with open(corpus_path, 'r') as fp:

            line_cnt = 0
            for line in fp.read().split('\n'):
                line_cnt += 1

                if line_cnt % 1000 == 0:
                    print(line_cnt)
                line = re.sub(Tokenizer.digits_pattern, Tokenizer.digits_placeholder, line)
                words = line.strip().split(delimiter)

                if len(line) == 0:
                    continue

                for ch in line:
                    if ch not in self.observations and ch != delimiter:
                        self.observations.append(ch)

                codes = reduce(Tokenizer.encode_reduce, words, [])

                # BEGIN DEBUG
                text = ''.join(words)
                assert len(text) == len(codes), "编码与句子不等长"
                # END DEBUG

                self.pi[codes[0]] += 1

                for x, y in zip(codes, codes[1:]):
                    self.A[x][y] += 1

                for state, observation in zip(codes, text):
                    if observation in self.B[state]:
                        self.B[state][observation] += 1
                    else:
                        self.B[state][observation] = 1

        # 初始状态矩阵频度转频率
        count = sum(self.pi.values())
        for state in self.pi.keys():
            self.pi[state] /= count

        # 转移矩阵频度转频率
        for state_from, state_to_dict in self.A.items():
            count = sum(state_to_dict.values())
            for (state_to, value) in state_to_dict.items():
                self.A[state_from][state_to] = value / count

        # 补全发射矩阵中
        for ch in self.observations:
            for state, character_dict in self.B.items():
                if ch not in character_dict:
                    character_dict[ch] = 1

        # 发射矩阵频度转频率
        for state, character_dict in self.B.items():
            count = sum(character_dict.values())
            for (character, frequency) in character_dict.items():
                self.B[state][character] = frequency / count


def test_sentence(sentence):
    tk = Tokenizer()
    print(tk.cut(sentence))


def test_all(corpus):
    tk = Tokenizer()
    inputfile = ''
    outputfile = ''
    if corpus == 'dev1':
        inputfile = './test/judge.data.1'
        outputfile = './test/result.hmm.1.txt'
    elif corpus == 'dev2':
        inputfile = './test/judge.data.2'
        outputfile = './test/result.hmm.2.txt'
    elif corpus == 'test1':
        inputfile = './test/final-test.1.txt'
        outputfile = './test/final-test.result.hmm.1.txt'
    elif corpus == 'test2':
        inputfile = './test/final-test.2.txt'
        outputfile = './test/final-test.result.hmm.2.txt'

    with open(inputfile, 'r', encoding='u8') as fin:
        with open(outputfile, 'w', encoding='u8') as fout:
            for line in fin.read().split('\n'):
                fout.write(tk.cut(line) + '\n')


if __name__ == "__main__":
    # test_sentence('淘宝网上的真皮卡包男式卡包牛皮男式卡包价格：￥138.00元，挺不错的，分享给大家，http://t.cn/a3S0o8')
    test_all('dev2')

