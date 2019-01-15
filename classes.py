from keras.preprocessing import sequence, text
import numpy as np
from unidecode import unidecode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

class Text_Sequence(object):
    
    def __init__(self, texto):
        self.texto=texto
    
# create mapping of unique words to integers  
    def creating_dict(self):
        words = text.text_to_word_sequence(self.texto, lower=True, split=" ", 
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\xe2\x80\x99\x98')

        unique_words = sorted(list(set(words)))
        len_words = len(unique_words)
        word_to_int = dict((c, i) for i, c in enumerate(unique_words))
        return word_to_int, len_words
        
        
# convert the words to integers using our lookup table we prepared earlier
    def mapping_to_dict(self, diccionario):
        dataX = []
        words = text.text_to_word_sequence(self.texto, lower=True, split=" ", 
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\xe2\x80\x99\x98')
        for i in words:
            dataX.append(diccionario[i])
            #converting the list of words into a numpy array
        words_array= np.asarray(dataX)
        return words_array
    
class Headlines(object):
    
    def __init__(self, periodicos, urls, titulares):
        self.periodicos=periodicos
        self.urls=urls
        self.titulares=titulares
       
    def keeping_min_headlines(self, df, min_num):
        journals_df = pd.DataFrame(columns=['Headline', 'Journal'])
        for k in self.periodicos:
            a = df.ix[(df['Journal']==k),].sample(n=min_num)
            d = [journals_df, a]
            journals_df = pd.concat(d)
        shuffled_headlines = journals_df.sample(frac=1).reset_index(drop=True)
        return shuffled_headlines
    
    def dataframing_headlines(self):
        headlines = []
        journal= []
        journal_names = dict(zip(self.urls, self.periodicos))
        for k,v in self.titulares.items():
            for i in v:
                journal.append(journal_names[k])
                headlines.append(i) 
        headlines_df = pd.DataFrame({'Headline': headlines, 'Journal': journal})
        shuffled_headlines = headlines_df.sample(frac=1).reset_index(drop=True)
        return shuffled_headlines
    
    def y_to_int(self, df):
        dict_periodicos = dict((c, i) for i, c in enumerate(self.periodicos))
        y_int = []
        for i in df.loc[:,"Journal"]:
            y_int.append(dict_periodicos[i])
        y_int = np.asarray(y_int)
        return y_int
    
    def int_to_journal(self, y):
        dict_periodicos = dict((i, c) for i, c in enumerate(self.periodicos))
        y_jour = []
        for i in y:
            y_jour.append(dict_periodicos[i])
        y_int = np.asarray(y_jour)
        return y_jour
    
#    def splitting_data(self, df, test_size, dictionary):
#        #x = df.loc[:,"Headline"].values
#        x = headlines_to_int(df, dictionary)
#        y = y_to_int(self, df)
#        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
#        return x_train, x_test, y_train, y_test

    def splitting_data(self, x, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def splitting_data_threesets(self, x, y, train_size):
        x_train, x_mix, y_train, y_mix = train_test_split(x, y, train_size=train_size)
        x_val, x_test, y_val, y_test = train_test_split(x_mix, y_mix, train_size=0.5)
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def concatenate_headlines(self, df):
        x = df.loc[:,"Headline"]
        delimiter = " "
        sentence = delimiter.join(x)
        sentence = unidecode(sentence)
        return sentence
        
    def headlines_to_int(self, df, dictionary):
        headlines_int = []
        for k in df.loc[:,"Headline"]:
            k = unidecode(k)
            palabras = text.text_to_word_sequence(k, lower=True, split=" ", 
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r')
            titular_int = []
            for i in palabras:
                titular_int.append(dictionary[i])
            headlines_int.append(titular_int)
            x_int = np.asarray(headlines_int)
         
        return x_int
        
    def max_hl_length(self, x_int):
        max_len = 0
        for x in x_int:
            if len(x)>max_len:
                max_len = len(x)
        print ('The longest headline consists of', max_len, 'words')
        return max_len
    
    def min_hl_number(self, df):
        sizes_journals = []
        for k in self.periodicos:
            num = df.ix[(df['Journal']==k), 'Headline'].count()
            sizes_journals.append(num)
            #if num<min_num:
            #    mim_num = num
            print(k, df.ix[(df['Journal']==k), 'Headline'].count())
        min_num = min(sizes_journals)
        return min_num
    
    def onehot_to_categorical(self, y):
        result = []
        for i in y:
            result.append(np.argmax(i))
        y_cat = np.asarray(result)
        return y_cat

class DL_model(object):
    
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def tuning_dl_model(self):
        # Step 1: tuning batch_size and epochs
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100, 150, 200]
        param_grid1 = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid1, iid=False, cv=5, scoring='neg_log_loss')
        grid_result = grid.fit(self.x_train, self.y_train)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        