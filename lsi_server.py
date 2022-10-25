import pandas as pd
import nltk
import numpy as np
import re
from collections import Counter
from nltk import ngrams
from nltk.stem.isri import ISRIStemmer
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
from nltk.corpus import stopwords
import os
import warnings
warnings.simplefilter("ignore")
import jieba
from gensim import corpora, models
from gensim.similarities import Similarity
from textblob import TextBlob
from tashaphyne.stemming import ArabicLightStemmer
import pyarabic.araby as araby
from flask import Flask
from flask import request
import json
 
ArListem = ArabicLightStemmer()


stop_word_comp = {"،","آض","آمينَ","آه","آهاً","آي","أ","أب","أجل","أجمع","أخ","أخذ","أصبح","أضحى","أقبل","أقل","أكثر","ألا","أم","أما","أمامك","أمامكَ","أمسى","أمّا","أن","أنا","أنت","أنتم","أنتما","أنتن","أنتِ","أنشأ","أنّى","أو","أوشك","أولئك","أولئكم","أولاء","أولالك","أوّهْ","أي","أيا","أين","أينما","أيّ","أَنَّ","أََيُّ","أُفٍّ","إذ","إذا","إذاً","إذما","إذن","إلى","إليكم","إليكما","إليكنّ","إليكَ","إلَيْكَ","إلّا","إمّا","إن","إنّما","إي","إياك","إياكم","إياكما","إياكن","إيانا","إياه","إياها","إياهم","إياهما","إياهن","إياي","إيهٍ","إِنَّ","ا","ابتدأ","اثر","اجل","احد","اخرى","اخلولق","اذا","اربعة","ارتدّ","استحال","اطار","اعادة","اعلنت","اف","اكثر","اكد","الألاء","الألى","الا","الاخيرة","الان","الاول","الاولى","التى","التي","الثاني","الثانية","الذاتي","الذى","الذي","الذين","السابق","الف","اللائي","اللاتي","اللتان","اللتيا","اللتين","اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","الى","اليوم","اما","امام","امس","ان","انبرى","انقلب","انه","انها","او","اول","اي","ايار","ايام","ايضا","ب","بات","باسم","بان","بخٍ","برس","بسبب","بسّ","بشكل","بضع","بطآن","بعد","بعض","بك","بكم","بكما","بكن","بل","بلى","بما","بماذا","بمن","بن","بنا","به","بها","بي","بيد","بين","بَسْ","بَلْهَ","بِئْسَ","تانِ","تانِك","تبدّل","تجاه","تحوّل","تلقاء","تلك","تلكم","تلكما","تم","تينك","تَيْنِ","تِه","تِي","ثلاثة","ثم","ثمّ","ثمّة","ثُمَّ","جعل","جلل","جميع","جير","حار","حاشا","حاليا","حاي","حتى","حرى","حسب","حم","حوالى","حول","حيث","حيثما","حين","حيَّ","حَبَّذَا","حَتَّى","حَذارِ","خلا","خلال","دون","دونك","ذا","ذات","ذاك","ذانك","ذانِ","ذلك","ذلكم","ذلكما","ذلكن","ذو","ذوا","ذواتا","ذواتي","ذيت","ذينك","ذَيْنِ","ذِه","ذِي","راح","رجع","رويدك","ريث","رُبَّ","زيارة","سبحان","سرعان","سنة","سنوات","سوف","سوى","سَاءَ","سَاءَمَا","شبه","شخصا","شرع","شَتَّانَ","صار","صباح","صفر","صهٍ","صهْ","ضد","ضمن","طاق","طالما","طفق","طَق","ظلّ","عاد","عام","عاما","عامة","عدا","عدة","عدد","عدم","عسى","عشر","عشرة","علق","على","عليك","عليه","عليها","علًّ","عن","عند","عندما","عوض","عين","عَدَسْ","عَمَّا","غدا","غير","ـ","ف","فان","فلان","فو","فى","في","فيم","فيما","فيه","فيها","قال","قام","قبل","قد","قطّ","قلما","قوة","كأنّما","كأين","كأيّ","كأيّن","كاد","كان","كانت","كذا","كذلك","كرب","كل","كلا","كلاهما","كلتا","كلم","كليكما","كليهما","كلّما","كلَّا","كم","كما","كي","كيت","كيف","كيفما","كَأَنَّ","كِخ","لئن","لا","لات","لاسيما","لدن","لدى","لعمر","لقاء","لك","لكم","لكما","لكن","لكنَّما","لكي","لكيلا","للامم","لم","لما","لمّا","لن","لنا","له","لها","لو","لوكالة","لولا","لوما","لي","لَسْتَ","لَسْتُ","لَسْتُم","لَسْتُمَا","لَسْتُنَّ","لَسْتِ","لَسْنَ","لَعَلَّ","لَكِنَّ","لَيْتَ","لَيْسَ","لَيْسَا","لَيْسَتَا","لَيْسَتْ","لَيْسُوا","لَِسْنَا","ما","ماانفك","مابرح","مادام","ماذا","مازال","مافتئ","مايو","متى","مثل","مذ","مساء","مع","معاذ","مقابل","مكانكم","مكانكما","مكانكنّ","مكانَك","مليار","مليون","مما","ممن","من","منذ","منها","مه","مهما","مَنْ","مِن","نحن","نحو","نعم","نفس","نفسه","نهاية","نَخْ","نِعِمّا","نِعْمَ","ها","هاؤم","هاكَ","هاهنا","هبّ","هذا","هذه","هكذا","هل","هلمَّ","هلّا","هم","هما","هن","هنا","هناك","هنالك","هو","هي","هيا","هيت","هيّا","هَؤلاء","هَاتانِ","هَاتَيْنِ","هَاتِه","هَاتِي","هَجْ","هَذا","هَذانِ","هَذَيْنِ","هَذِه","هَذِي","هَيْهَاتَ","و","و6","وا","واحد","واضاف","واضافت","واكد","وان","واهاً","واوضح","وراءَك","وفي","وقال","وقالت","وقد","وقف","وكان","وكانت","ولا","ولم","ومن","مَن","وهو","وهي","ويكأنّ","وَيْ","وُشْكَانََ","يكون","يمكن","يوم","ّأيّان"}
stops = set(stopwords.words("arabic"))

def tokenizer(text):
    item_str = jieba.lcut(text,cut_all=True)
    while ' 'in item_str:
        item_str.remove(' ')
    while ''in item_str:
        item_str.remove('')
    return item_str


def preprocess(text, flags=[1,1,1,1,1,1]):
    def remove_number(text, work=1):
 
        if work==0:
            return text
        s=text
        numberP="(\d+\.\d+)"
        numbers=re.findall(numberP,s)
        for number in numbers:
            s=s.replace(str(number),'')
        numberP="\d+"
        numbers=re.findall(numberP,s)
        for number in numbers:
            s=s.replace(str(number),'')
        return s
    def remove_metacharachters(inText,work=1):
        text=''
        if work==0:
            return inText
        if work==1:
            text=re.sub(r"\.|,|\'|!|\?|!|@|#|\$|\%|\^|\&|\*|_|\+|-|=|<|>|:|\(|\)|\\|/|\n|،|×|\"|؛|x","",inText)
        return text
    def remove_stop_words(text,work=1):
        if work==0:
            return text
        zen = TextBlob(text)
        words = zen.words
        return " ".join([w for w in words if not w in stops and not w in stop_word_comp and len(w) >= 2])
    def normalizeArabic(text,work=1):
        if work==0:
            return text
        text = text.strip()
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        noise = re.compile(""" ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        text = re.sub(noise, '', text)
        text = re.sub(r'(.)\1+', r"\1\1", text) # Remove longation
        return araby.strip_tashkeel(text)
    def standardization(tweet,work=1):
        words = tweet.split()
        result = []
        if work ==1 :
            stemmer = ISRIStemmer()
            for word in words:
                word = stemmer.norm(word, num=1) 
                if word.startswith('#'):
                    result.append(word)
                    continue
                if not word in stemmer.stop_words:    
                    word = stemmer.pre32(word)         
                    word = stemmer.suf32(word)        
                result.append(word)
        if work != 1:
            result=words
        return ' '.join(result)
    def stem(text,work=1):
        if(work==0):
            return text
        zen = TextBlob(text)
        words = zen.words
        cleaned = list()
        for w in words:
            ArListem.light_stem(w)
            cleaned.append(ArListem.get_root())
        return " ".join(cleaned)
    text =remove_number(text, flags[0])
    text = remove_metacharachters(text, flags[1])
    text = remove_stop_words(text, flags[2])
    text = normalizeArabic(text, flags[3])
    text = standardization(text, flags[4])
    text=  stem(text,flags[5])
    return text

def changeToList(mmcorpus):
        newList=[]
        for l in mmcorpus:
            il=[]
            for item in l:
                il.append(item)
            newList.append(il)
        return newList

app = Flask(__name__)
@app.route('/search', methods=['GET'])
def search():
    
    quary=request.form.get('quary')
    threshold=float(request.form.get('threshold'))
    max_number=int(request.form.get('max_number'))
    
    dictionary=corpora.Dictionary.load('dictionary.txt')
    corpus = corpora.MmCorpus ('corpuse.mm')
    tfidf_model=models.TfidfModel(corpus)
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    lsi= models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=50)
    corpus_lsi = [lsi[doc] for doc in corpus]
    similarity_lsi=Similarity('Similarity-Lsi-index', corpus_lsi, num_features=400,num_best=max_number)  
    def find(test):
        test=preprocess(test)
        test_cut_raw_1 =tokenizer(test)
        test_corpus_3 = dictionary.doc2bow(test_cut_raw_1) 
        test_corpus_tfidf_3 = tfidf_model[test_corpus_3]  
        test_corpus_lsi_3 = lsi[test_corpus_tfidf_3] 
        simms=similarity_lsi[test_corpus_lsi_3]
        return simms
    ids=[]
    with open("ids.txt") as file:
        for line in file:
            ids.append(int(line.rstrip()))
    simms=find(quary)
    result=[]
    for x in simms:
        if x[1]>= threshold:  
            result.append(ids[x[0]] )
    return json.dumps({'id':result})
@app.route('/add', methods=['POST'])
def add():
    description=request.form.get('description')
    categories=request.form.get('categories')
    Id=int(request.form.get('id'))
    
        
    s=description+' '+categories
    corpus = corpora.MmCorpus ('corpuse.mm')
    corpus =changeToList(corpus)
    dictionary=corpora.Dictionary.load('dictionary.txt')
    clean=preprocess(s)
    tokens =tokenizer(clean)
    dictionary.add_documents([tokens])
    dictionary.save('dictionary.txt')
    corpus.append(dictionary.doc2bow(tokens))
    corpora.MmCorpus.serialize ('corpuse.mm', corpus)
    textfile = open("ids.txt", "a")
    textfile.write(str(Id)+ "\n")
    textfile.close()
    return json.dumps('success')
@app.route('/delete', methods=['DELETE'])
def delete():
    Id=int(request.form.get('id'))
    corpus = corpora.MmCorpus ('corpuse.mm')
    corpus =changeToList(corpus)
    ids=[]
    with open("ids.txt") as file:
        for line in file:
            ids.append(int(line.rstrip()))
    
    oneD_array=np.array(ids)
    index, = np.where(oneD_array == Id)
    if len(index)>0:
        del ids[index[0]]
        del corpus[index[0]]
    corpora.MmCorpus.serialize ('corpuse.mm', corpus)
    
    textfile = open("ids.txt", "w")
    for element in ids:
        textfile.write(str(element)+ "\n")
    textfile.close()
    return json.dumps('success')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
    