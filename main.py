from hazm import word_tokenize, Normalizer
import numpy as np
import os, string

# Persian stop words to be excluded from the document 
stop_words = ['دیگران', 'همچنان', 'مدت', 'چیز', 'سایر', 'جا', 'طی', 'کل', 'کنونی', 'بیرون', 'مثلا', 'کامل', 'کاملا', 'آنکه', 'موارد', 'واقعی',
              'امور', 'اکنون', 'بطور', 'بخشی', 'تحت', 'چگونه', 'عدم', 'نوعی', 'حاضر', 'وضع', 'مقابل', 'کنار', 'خویش', 'نگاه', 'درون', 'زمانی',
              'بنابراین', 'تو', 'خیلی', 'بزرگ', 'خودش', 'جز', 'اینجا', 'مختلف', 'توسط', 'نوع', 'همچنین', 'آنجا', 'قبل', 'جناح', 'اینها', 'طور',
              'شاید', 'ایشان', 'جهت', 'طریق', 'مانند', 'پیدا', 'ممکن', 'کسانی', 'جای', 'کسی', 'غیر', 'بی', 'قابل', 'درباره', 'جدید', 'وقتی', 'اخیر',
              'چرا', 'بیش', 'روی', 'طرف', 'جریان', 'زیر', 'آنچه', 'البته', 'فقط', 'چیزی', 'چون', 'برابر', 'هنوز', 'بخش', 'زمینه', 'بین', 'بدون',
              'استفاده', 'همان', 'نشان', 'بسیاری', 'بعد', 'عمل', 'روز', 'اعلام', 'چند', 'آنان', 'بلکه', 'امروز', 'تمام', 'بیشتر', 'آیا', 'برخی', 'علیه',
              'دیگری', 'ویژه', 'گذشته', 'انجام', 'حتی', 'داده', 'راه', 'سوی', 'ولی', 'زمان', 'حال', 'تنها', 'بسیار', 'یعنی', 'عنوان', 'همین', 'هبچ',
              'پیش', 'وی', 'یکی', 'اینکه', 'وجود', 'شما', 'پس', 'چنین', 'میان', 'مورد', 'چه', 'اگر', 'همه', 'نه', 'دیگر', 'آنها', 'باید', 'هر', 'او',
              'ما', 'من', 'تا', 'نیز', 'اما', 'یک', 'خود', 'بر', 'یا', 'هم', 'را', 'این', 'با', 'آن', 'برای', 'و', 'در', 'به', 'که', 'از'
              'کن', 'کرد', 'کردن', 'باش', 'بود', 'بودن', 'شو', 'شد', 'شدن', 'ددار', 'داشت', 'داشتن', 'خواه', 'خواست', 'خواستن', 'گوی', 'گفت',
              'گفتن', 'گرفت', 'گرفتن', 'آمد', 'آمدن', 'توانست', 'توانستن', 'یافت', 'یافتن', 'آورد', 'آوردن','هرگز','نمي کند', 'است','هستند','با','از','چه','باشد',
              'مي کنند']

def remove_punctuation(txt):
    s = ""
    punc = string.punctuation+'؟،؛«»'
    for l in txt:
        if l not in punc:
            s+=l
    return s

def document(filepath):
    f = open(filepath, 'r', encoding='utf-8', errors='ignore')
    txt = f.read()
    f.close()

    txt = remove_punctuation(txt)
    
    normalizer = Normalizer()
    txt = normalizer.normalize(txt)
    
    document = word_tokenize(txt)
    
    document = [word for word in document if word not in stop_words and not word.isdigit()]
    
    return document


#builds a list of all documents
documents = []
files = os.listdir(r'./documents')
for file in files:
    documents.append(document(r'./documents/'+file))

number_of_documents = len(files)

#builds a corpus
words=[]
for i in documents:
    words += i
words = list(set(words))
number_of_words = len(words)

#construction of the bag of words matrix
bag_of_words  = np.zeros((number_of_documents, number_of_words))
for i in range(number_of_documents):
    for j in range(number_of_words):
        bag_of_words[i, j] = documents[i].count(words[j])

#number of clusters is set to be 3
K = 3

def calculate_pzdw(pwz, pzd):
    number_of_words, K = pwz.shape
    number_of_documents = pzd.shape[1]
    
    pzdw = np.zeros((K, number_of_words, number_of_documents))

    for i in range(number_of_documents):
            pzdw[:, :, i] = (pwz * pzd[:, i]).T

    for i in range(number_of_documents):
            denom = pzdw[:, :, i].sum(axis=0)
            pzdw[:, :, i] /= denom

    return pzdw

def calculate_pwz(pzdw, bag_of_words):
    K, number_of_words, number_of_documents = pzdw.shape

    pwz = np.zeros((number_of_words, K))
 
    for j in range(number_of_words):
        pwz[j] = (bag_of_words[:, j] * pzdw[:, j, :]).sum(axis=1)

    for k in range(K):  
        denom = (bag_of_words * pzdw[k].T).sum()
        pwz[:, k] /= denom

    return pwz

def calculate_pzd(pzdw, bag_of_words):
    K, number_of_words, number_of_documents = pzdw.shape

    pzd = np.zeros((K, number_of_documents))

    for k in range(K):
        pzd[k] = (bag_of_words * pzdw[k].T).sum(axis=1)

    for k in range(K):
        pzd[k] /= bag_of_words.sum(axis=1)

    return pzd

def plsa(bag_of_words, K, number_of_iterations=10, epsilon=0.0001):
    number_of_documents, number_of_words = bag_of_words.shape
    
    # pwz and pzd are randomly initialized
    pwz = np.random.rand(number_of_words, K)
    pzd = np.random.rand(K, number_of_documents)

    # matrices are normalized to attain probabilities 
    normalize = pwz.sum(axis=0)
    pwz /= normalize

    normalize = pzd.sum(axis=0)
    normalize.shape = (1, number_of_documents)
    pzd /= normalize

    for i in range(number_of_iterations):
        last_pwz = np.copy(pwz)

        #E step
        pzdw = calculate_pzdw(pwz, pzd)

        #M step
        pwz = calculate_pwz(pzdw, bag_of_words)
        pzd = calculate_pzd(pzdw, bag_of_words)

        pwz_change = ((last_pwz - pwz) ** 2).sum()

        if pwz_change < epsilon:
            break
        
    return pwz, pzd


pwz, pzd = plsa(bag_of_words, K, 1000)
for k in range(K):
    print ("class: ", k)
    terms = sorted(enumerate(pwz[:, k]), key=lambda x: x[1], reverse=True)
    for term_id, score in terms[:20]:
        print("{word:25}{score:45}".format(score=str(score),word=words[term_id]))

