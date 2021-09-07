class Model():
    
    def __init__(self, file, norm=True):
        '''
        Model of Classifing massages
        imput: file -- name of file (str)
               norm -- using stemming (boolean)
        '''
        
        # считываем данные из фала
        self.df_train = pd.read_excel(file, sheet_name='Training data')
        self.df_test = pd.read_excel(file, sheet_name='Test data')
        
        # объединяем датасеты
        self.df = pd.concat([self.df_train, self.df_test], ignore_index=True)
        
        #
        if norm == True:
            self.train_text = self.stemming(self.df_train['Пример текста'])
            self.test_text = self.stemming(self.df_test['Пример текста'])
            self.text = self.stemming(self.df['Пример текста'])
        else:
            self.train_text = self.df_train['Пример текста']
            self.test_text = self.df_test['Пример текста']
            self.text = self.df['Пример текста']
        
        #print(len(self.train_text), len(self.test_text), len(self.df))
        
        self.train_target = self.df_train['Класс']
        self.test_target = self.df_test['Класс']
        
        self.categories = []
        for t in self.train_target:
            if t not in self.categories:
                print(t)
                self.categories.append(t)
        
        
        self.x_train, self.x_test = self.text_to_matrix(self.text)
        #print(self.x_train, self.x_test)
        
        self.y_train = self.make_num_target(self.train_target)
        self.y_test = self.make_num_target(self.test_target)
        
        self.feauter_names = self.count.get_feature_names()
        print('Количество признаков: {}'.format(len(feauter_names)))
        
    def fit(self):
        '''
        Lear the model
        '''
        # один против остальных
        from sklearn.svm import LinearSVC
        self.linear_svm = LinearSVC()
        self.linear_svm.fit(x_train, y_train)
        print('Правильность на обучающем наборе: ', linear_svm.score(x_train, y_train))
        print('Правильность на тестовом наборе: ', linear_svm.score(x_test, y_test))
        self.predictions = self.linear_svm.predict(x_test)
        
    def predict(self, text):
        '''
        Make a prediction
        '''
        text = self.stemming(pd.Series([text]))
        x = self.count.transform(text)
        answer = self.linear_svm.predict(x)[0]
        print('Предсказание:', self.categories[answer])
        return answer
        
    def presentation(self):
        for text, p, y in zip(self.test_text, self.predictions, self.y_test):
            if p == y:
                r = 'Right'
            else:
                r = 'Wrong'
            print(text, self.categories[p], r)
        
    def stemming(self, texts):
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('russian')
        new_texts = []
        for t in texts:
            words = t.split(' ')
            tt = ' '.join([stemmer.stem(word) for word in words])
            new_texts.append(tt)
        #print(len(new_texts))
        return pd.Series(new_texts)
                
    def text_to_matrix(self, text):
        from sklearn.feature_extraction.text import CountVectorizer
        self.count = CountVectorizer()
        X = self.count.fit_transform(text)
        self.voc = self.count.vocabulary_
        x = X.toarray()
        x_train = x[0:len(self.train_text),:]
        #print(len(self.train_text), len(self.df))
        x_test = x[len(self.train_text):len(self.train_text)+len(self.test_text),:]
        return x_train, x_test
        
    def make_num_target(self, target):
        y = []
        for t in target:
            for n in range(0, len(self.categories)):
                if t == self.categories[n]:
                    y.append(n)           
        return np.array(y)
    
model = Model('Test.xlsx', norm=True)
model.fit()
model.predict('я врач и хочу зарплату и отдыхать')
