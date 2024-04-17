import pandas as pd
import re
from collections import Counter
from math import log10
from itertools import product


class Base:
    def __init__(self, word, text):
        self._real_text = text.split('\n\n')
        self._docs = self.__create_lists(re.sub(r'[.?,!<>():;]', '', text))
        self._len_of_docs = len(self._docs)
        self._word = Counter(self.__create_lists(word, ' ')).keys()
        self._df = None

    @staticmethod
    def __create_lists(z, separator='\n\n'):
        return list(map(lambda x: x.strip(' ?\n\t.,;:').lower(), z.split(separator)))

    @property
    def real_text(self):
        return self._real_text

    def information(self):
        print(f'\nКлючевые слова: {', '.join(self._word)}\n'
              f'Количество docs: {self._len_of_docs}\n\n'
              f'{self._df}')

    def check_df(self):
        return self._df


class CreateA(Base):
    def __init__(self, word, text):
        super().__init__(word, text)

    def calculate_tf_and_idf(self):
        data_tf, data_idf = {}, {}
        for x in self._word:
            data_tf['TF_' + x] = [Counter(i.split())[x] / len(i.split()) for i in self._docs]
            count_of_word_in_docs = sum(map(bool, data_tf['TF_' + x]))
            if count_of_word_in_docs:
                data_idf['IDF_' + x] = [log10(self._len_of_docs / sum(map(bool, data_tf['TF_' + x]))) for _ in range(len(self._docs))]
            else:
                data_idf['IDF_' + x] = 0

        data_tf.update(data_idf)
        self._df = pd.DataFrame(data_tf)
        self.__calculate_full_df()
        return self

    def __calculate_full_df(self):
        for x in self._word:
            self._df['TF-IDF_' + x] = self._df['TF_' + x] * self._df['IDF_' + x]
        self._df['TF-IDF'] = sum(self._df['TF-IDF_' + x] for x in self._word)
        # print(self.__df)

    def sort_by_tf_idf(self):
        self._df.sort_values('TF-IDF', ascending=False, inplace=True)
        return self

    def describe_df(self):
        # print(self.__df)
        for i in self._df.head().index:
            print(self._real_text[i])


class CreateB(Base):
    def __init__(self, word, text):
        super().__init__(word, text)
        self.__configurate_docs()
        # print(self._docs)

    def __configurate_docs(self):
        list_of_kw = []
        for z in self._docs:
            data = []
            for word in self._word:
                doc = z.split()
                data.append([i for i in range(len(doc)) if doc[i] == word])
            list_of_kw.append(data)
        #print(list_of_kw)
        self._df = list_of_kw

    def make_tuples(self):
        list_for_sort = []
        for d in self._df:
            if all([bool(i) for i in d]):
                #print(*map(lambda x: tuple(sorted(x)), product(*d)))
                mas_sum = []
                for p in map(lambda x: tuple(sorted(x)), product(*d)):
                    mas_sum.append(sum([p[j] - p[j-1] for j in range(len(p) - 1, 0, -1)]))
                list_for_sort.append(min(mas_sum))
            else:
                list_for_sort.append(self._len_of_docs)
        self._df = pd.DataFrame(list_for_sort)
