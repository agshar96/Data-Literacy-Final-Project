'''

A. The top books of each year are extracted on the basis of two sources of GoodReads.com
1. "Meta Data of Books" archive extracted from UCSD Book Graph

   https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK

2. Data Scraping performed on https://www.goodreads.com/, getting the information of the books from https://en.wikipedia.org/wiki/Lists_of_The_New_York_Times_nonfiction_best_sellers and 
https://en.wikipedia.org/wiki/Lists_of_The_New_York_Times_fiction_best_sellers

B. A test is performed to understand the similarities of books in order to categorize them into genres.
   Although this is not applied to the actual dataset, it helps in adding the final touches to assigning the correct genres
   to the books, which was to be done manually (data validation).

'''

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set_theme(style='whitegrid')
import matplotlib.pyplot as plt

import re
import requests
import json

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


'''
A.
Source 1 : ISBN mapping using Open Library API to get the json of each book (containing information)
'''

list_of_books =     \
   ['book1-100k.csv',              
    'book300k-400k.csv',
    "book1000k-1100k.csv",            
    "book100k-200k.csv",           
    "book1100k-1200k.csv",          
    "book1200k-1300k.csv",         
    'book1300k-1400k.csv',        
    "book1400k-1500k.csv",       
    "book1500k-1600k.csv",      
    "book1600k-1700k.csv",     
    "book1700k-1800k.csv",    
    "book1800k-1900k.csv", 
    "book1900k-2000k.csv",  
    "book2000k-3000k.csv",
    "book200k-300k.csv",
    "book3000k-4000k.csv",
    "book4000k-5000k.csv",
    "book400k-500k.csv",
    "book500k-600k.csv",
    "book600k-700k.csv",
    "book700k-800k.csv",
    "book800k-900k.csv",
    "book900k-1000k.csv",
    ]

cols=['Id', 'Name', 'Authors', 'ISBN', 'Rating', 'PublishYear',
       'PublishMonth', 'PublishDay', 'Publisher', 'RatingDist5', 'RatingDist4',
       'RatingDist3', 'RatingDist2', 'RatingDist1', 'RatingDistTotal',
       'CountsOfReview', 'Language', 'pagesNumber', 'Description',
       'Count of text reviews']
res = pd.DataFrame(columns=cols)
authors = []

# iterating over each file and picking the top 20 most rated books of every year
# note : each file consists around 100k books
for b in list_of_books:
    df = pd.read_csv(f'archive/{b}')
    df = df[(df.Language.isin(['eng','en-US','en-GB']))]      
    df['Name_'] = df['Name'].apply(lambda x: re.sub(r'\W+', ' ',x))
    df['Authors_'] = df['Authors'].apply(lambda x: re.sub(r'\W+', ' ',x))
    df.drop_duplicates(subset=['Authors_'],inplace=True)
    df.drop_duplicates(subset=['Name_'],inplace=True) 
    df['RatingDistTotal'] = df['RatingDistTotal'].apply(lambda x: int(x.split('total:')[-1]))
    df['RatingDist5'] = df['RatingDist5'].apply(lambda x: int(x.split('5:')[-1]))
    df['RatingDist4'] = df['RatingDist4'].apply(lambda x: int(x.split('4:')[-1]))
    df['RatingDist3'] = df['RatingDist3'].apply(lambda x: int(x.split('3:')[-1]))
    df['RatingDist2'] = df['RatingDist2'].apply(lambda x: int(x.split('2:')[-1]))
    df['RatingDist1'] = df['RatingDist1'].apply(lambda x: int(x.split('1:')[-1]))
    
    filt_df = df[df.PublishYear.isin(years)].sort_values(by=['PublishYear','RatingDistTotal'],
                                           ascending=False).groupby('PublishYear').head(20)
    res = pd.concat([res,filt_df],ignore_index=True)
    authors.append(list(df['Authors_'].unique()))
    
res.drop_duplicates(subset=['Name_','Authors_'],inplace=True)
res.drop_duplicates(subset=['Authors_','Rating'],inplace=True)

# keeping just the top 10 books of every year
res = res.sort_values(by=['PublishYear','RatingDistTotal'],
                      ascending=False).groupby('PublishYear').head(10).reset_index(drop=True)

# removing unnecessary columns
data = res[['Name',
            'Authors',
            'Rating', 
            'ISBN',
            'PublishYear',
             'RatingDist5', 'RatingDist4',
             'RatingDist3', 'RatingDist2', 'RatingDist1', 'RatingDistTotal',
            'CountsOfReview','pagesNumber','PagesNumber'
           ]]
data['no_pages'] = data.pagesNumber.combine_first(data.PagesNumber)
data.drop(['pagesNumber','PagesNumber'],axis=1,inplace=True)


# genre assignment - genre 1 and genre 2 using the book's ISBN
for index, row in data.iterrows():
    try:
        ISBN = row['ISBN']
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{ISBN}&format=json&jscmd=data"
        response = requests.get(url)
        text = json.loads(response.text)[f"ISBN:{ISBN}"]
        print(text['title'],row['PublishYear'])
        genre1 = text['subjects'][0]['name']
        genre2 = text['subjects'][1]['name']              
    except:
        print("error",row['ISBN'])
        genre1 = genre2 = np.nan
        
    data.at[index, 'genre1'] = genre1
    data.at[index, 'genre2'] = genre2

    
# save book dataset
data1 = data.copy()
data1.to_csv("book_data_1.csv")


'''
Source 2 : Web Scraping on https://www.goodreads.com/
'''

best_sellers = pd.read_excel("best_sellers_book_list.xlsx") # list of best sellig books taken from wikipedia
best_sellers = best_sellers[['title','author','year']].copy()
best_sellers['author'] = best_sellers['author'].apply(lambda x: remove_sq_brackets(x))
best_sellers['book author'] = best_sellers['title']+" "+best_sellers['author']
year = {i:j for i,j in zip(best_sellers['book author'],best_sellers['year'])}

# Scraping tool used - selenium
books_table = pd.DataFrame(columns=['Year','Actual Book','Displayed Book','Author','Rating','No_Ratings','Pages','Genre'])
chrome_options = webdriver.ChromeOptions(); 
chrome_options.add_experimental_option("excludeSwitches", ['enable-automation']);
chromedriver = Service('/Users/eashwar/Documents/Projects/Book Analysis/chromedriver')
driver = webdriver.Chrome(service=chromedriver, options=chrome_options)

for b,a in zip(best_sellers['book author'],best_sellers['title']):
    driver.get('https://www.goodreads.com/search')
    driver.maximize_window()
    driver.find_element(By.ID,'search_query_main').send_keys(b)
    driver.find_element(By.CLASS_NAME,"searchBox__button").click()
    driver.refresh()

    try:
        displayed_book = driver.find_element(By.CLASS_NAME,"bookTitle").text
        mini = driver.find_element(By.CLASS_NAME,"minirating").text
        driver.find_element(By.CLASS_NAME,"bookTitle").click()
        try:
            author = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[4]/div[1]/div[2]/div[1]/span[2]/div/a/span").text
        except:
            author = "-"
        try:
            genre = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[5]/div[7]/div/div[2]/div").text
        except:
            try:
                genre = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[5]/div[6]/div/div[2]/div/div[1]/div[1]/a").text
            except:
                try:
                    genre = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/div[1]/div[2]/div[1]/div[2]/div[6]/ul/span[1]/span[2]").text
                except:
                    genre = np.nan

        try:
            no_ratings = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[4]/div[1]/div[2]/div[2]/a[2]").text
        except:
            no_ratings = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/div[1]/div[2]/div[1]/div[2]/div[2]/a/div[2]/div/span[1]").text

        try:
            no_pages = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[4]/div[1]/div[2]/div[5]/div[1]").text
        except:
            no_pages = driver.find_element(By.CLASS_NAME,"FeaturedDetails").text
        try:
            rating = driver.find_element(By.XPATH,"/html/body/div[2]/div[3]/div[1]/div[2]/div[4]/div[1]/div[2]/div[2]/span[2]").text
        except:
            rating = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/div[1]/div[2]/div[1]/div[2]/div[2]/a/div[1]/div").text

        new_row = {'Year':year[b],
                   'Actual Book':a,
                   'Displayed Book':displayed_book,
                   'Author':author,
                   'Rating':rating,
                   'No_Ratings':no_ratings,
                   'Pages':no_pages,
                   'Genre':genre}
        print(new_row)
        books_table = books_table.append(new_row,ignore_index=True)
    except:
        pass

driver.quit()

# save book dataset
data2 = books_table.copy()
data2.to_csv("book_data_2.csv")


'''
B. Data validation - to identify the correct genres
1. Clustering the dataset using K-Means in order to categorize the genre from the 'genre1' and 'genre2'
'''


def removeDuplicatesFromText(text):
    regex = r'\b(\w+)(?:\W+\1\b)+'
    return re.sub(regex, r'\1', text, flags=re.IGNORECASE)


def process_(x):
    x = x.lower()
    x = x.strip()
    x = re.sub(r'\W+', ' ', x)
    x = [word.strip() for word in x.split(" ") if word not in set(stopwords.words('english'))]

    x = ' '.join(x)
    x = removeDuplicatesFromText(x)
    x = ''.join([i for i in x if not i.isdigit()])
    words = word_tokenize(x)
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        if word not in ['fiction', 'book']:
            lemmatized_words.append(lemmatizer.lemmatize(word))

    x = ' '.join(lemmatized_words)

    return x


def generate_N_grams(text, ngram=1):
    text = text.lower().strip()
    words = [word.strip() for word in text.split(" ") if word not in set(stopwords.words('english'))]
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans

temp = data2.copy()
temp['genre1'] = temp['genre1'].replace(np.nan,' ').apply(process_)
temp['genre2'] = temp['genre2'].replace(np.nan,' ').apply(process_)
temp['genre'] = temp['genre1']+' '+temp['genre2']
temp['genre'] = temp['genre'].apply(process_)

documents = temp['genre2'].values
vectorizer = TfidfVectorizer(stop_words='english')

try:
    features = vectorizer.fit_transform(documents)
except Exception as e:
    print(e)
    pass

k = 6
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

temp['cluster'] = model.labels_
temp=temp.sort_values(by='cluster').reset_index(drop=True) # a cluster is now assigned to each book as a genre


''' 
2. Checking Book similarities using Levenshtein scores 
(to understand how closely a genre could be assigned to each book)
'''
from distance import levenshtein

# Create the dataframe with sample books
books = pd.DataFrame(
    {'title': ["Anne Frank: The Diary of a Young Girl", "The Diary of Anne Frank", "The Diary of a Young Girl"]})

# Create an empty dataframe to store the similarity scores
similarity_scores = pd.DataFrame(columns=books['title'], index=books['title'])

# Iterating through the titles and calculating the Levenshtein similarity scores
for i in range(len(books)):
    for j in range(i + 1, len(books)):
        score = levenshtein(books.iloc[i]['title'],
                            books.iloc[j]['title'])
        similarity_scores.at[books.iloc[i]['title'], books.iloc[j]['title']] = score

print(similarity_scores)


