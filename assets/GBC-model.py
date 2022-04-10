from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
from pymystem3 import Mystem
from string import punctuation
from joblib import dump, load


def make_predict(report_text):
    clf = load('GradientBoostingModel.joblib')  
    russian_stopwords = ['и', 'в', 'во', 'не', 'что','он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты',
     'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня' 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
     'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
     'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
     'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
     'здесь', 'этом', 'один', 'почти', 'мой',
 'тем',
 'чтобы',
 'нее',
 'сейчас',
 'были',
 'куда',
 'зачем',
 'всех',
 'никогда',
 'можно',
 'при',
 'наконец',
 'два',
 'об',
 'другой',
 'хоть',
 'после',
 'над',
 'больше',
 'тот',
 'через',
 'эти',
 'нас',
 'про',
 'всего',
 'них',
 'какая',
 'много',
 'разве',
 'три',
 'эту',
 'моя',
 'впрочем',
 'хорошо',
 'свою',
 'этой',
 'перед',
 'иногда',
 'лучше',
 'чуть',
 'том',
 'нельзя',
 'такой',
 'им',
 'более',
 'всегда',
 'конечно',
 'всю',
 'между',
 'это',
 'также',
 'данный',
 'слово',
 'который']
    no_features = 40

    dataset_to_extract_features = pd.read_csv('assets/augmented-dataset.csv', on_bad_lines='skip')

    # with open('assets/stopwords.csv', encoding = 'utf-8') as csvfile:
    #     russian_stopwords = list(csv.reader(csvfile))
    #     print(russian_stopwords)
    

    # russian_stopwords = pd.read_csv('assets/stopwords.csv')

    text_for_features = dataset_to_extract_features[' текст_обращения']
    text_for_features = text_for_features.append(pd.Series([report_text]))

    tfidf_vectorizer = TfidfVectorizer(max_features=no_features, min_df=1, stop_words=russian_stopwords)

    text_to_predict = tfidf_vectorizer.fit_transform(text_for_features).toarray()

    new_text_to_predict = text_to_predict[-1]
    new_text_to_predict = [new_text_to_predict]
    # print(new_text_to_predict)

    # text_to_predict = tfidf_vectorizer.fit_transform([report_text]).toarray()
    pr = clf.predict(new_text_to_predict)
    return pr

# random, 0
# sample_text = 'претензия хотеть клиент  –  нужно срочно попадать прием терапевт ничто толком объяснять сотрудник ничто хотеть решать отправлять пациент просить решение вопрос обратный связь'
# defined, 1
# sample_text = 'доводить ваш сведение горячий линия обращаться обращаться пз 8.30 принимать анализ антитело коронавирус т 1 регистратор большой очередь клиент указывать несоблюдение социальный дистанция регистратор кашель'
# defined, 3
sample_text = 'доводить ваш сведение горячий линия обращаться пациент пациент обращаться пз краснодар примерно 13 14 час грудной ребенок сдача анализ грубый форма сказать медсестра сдавать анализ необходимо утро т момент обращение сдавать анализ ковид пациент возмущать форма отказ'
print(make_predict(sample_text))