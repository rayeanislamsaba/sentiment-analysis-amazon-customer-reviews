#Basic libraries
import pandas as pd 
import numpy as np 

#NLTK libraries
import nltk
import re
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Machine Learning libraries
import sklearn 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn import preprocessing

#Metrics libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Other miscellaneous libraries
from itertools import cycle
from collections import defaultdict
from collections import Counter
from imblearn.over_sampling import SMOTE

reviews = pd.read_csv(r"E:\Vs py\Sentiment Analysis\amazon_reviews.csv")

print ("The shape of the  data is (row, column):"+ str(reviews.shape))
print (reviews.info())
print (reviews.head())

#Creating a copy
process_reviews=reviews.copy()

#Checking for null values
process_reviews.isnull().sum()

process_reviews['reviewText']=process_reviews['reviewText'].fillna('Missing')
process_reviews['reviews']=process_reviews['reviewText']+process_reviews['summary']
process_reviews=process_reviews.drop(['reviewText', 'summary'], axis=1)
print(process_reviews.head())

#Figuring out the distribution of categories
print(process_reviews['overall'].value_counts())

#PIE CHART 
fig = plt.figure(figsize=(7, 7))
colors = ('red', 'green', 'blue', 'orange', 'yellow')
wp = {'linewidth': 1, "edgecolor": 'black'}
tags = process_reviews['overall'].value_counts(normalize=True)
explode = (0.1,) * len(tags) 

tags.plot(kind='pie',
          autopct="%1.1f%%",
          shadow=True,
          colors=colors[:len(tags)],
          startangle=90,
          wedgeprops=wp,
          explode=explode,
          label='Rating Distribution (%)')

plt.title("Percentage-wise Distribution of Ratings")
plt.ylabel('')  

# Save to memory if needed
graph = BytesIO()
fig.savefig(graph, format="png")
plt.show()
def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['overall'] == 3.0:
        val = 'Neutral'
    elif row['overall'] == 1.0 or row['overall'] == 2.0:
        val = 'Negative'
    elif row['overall'] == 4.0 or row['overall'] == 5.0:
        val = 'Positive'
    else:
        val = -1
    return val
#Applying the function in the new column
process_reviews['sentiment'] = process_reviews.apply(f, axis=1)
print(process_reviews.head())
print(process_reviews['sentiment'].value_counts())

#new data frame which has date and year
new = process_reviews["reviewTime"].str.split(",", n = 1, expand = True) 
  
#making separate date column from new data frame 
process_reviews["date"]= new[0] 
  
#making separate year column from new data frame 
process_reviews["year"]= new[1] 

process_reviews=process_reviews.drop(['reviewTime'], axis=1)
print(process_reviews.head())

#Splitting the date 
new1 = process_reviews["date"].str.split(" ", n = 1, expand = True) 
  
#adding month to the main dataset 
process_reviews["month"]= new1[0] 
  
#adding day to the main dataset 
process_reviews["day"]= new1[1] 

process_reviews=process_reviews.drop(['date'], axis=1)
print (process_reviews.head())

#Splitting the dataset based on comma and square bracket 
new1 = process_reviews["helpful"].str.split(",", n = 1, expand = True)
new2 = new1[0].str.split("[", n = 1, expand = True)
new3 = new1[1].str.split("]", n = 1, expand = True)

#Resetting the index
new2.reset_index(drop=True, inplace=True)
new3.reset_index(drop=True, inplace=True)

#Dropping empty columns due to splitting 
new2=new2.drop([0], axis=1)
new3=new3.drop([1], axis=1)

#Concatenating the splitted columns
helpful=pd.concat([new2, new3], axis=1)

#few spaces in new3, so it is better to strip all the values to find the rate
def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

helpful= trim_all_columns(helpful)

#Converting into integer types
helpful[0]=helpful[0].astype(str).astype(int)
helpful[1]=helpful[1].astype(str).astype(int)

#Dividing the two columns
try:
  helpful['result'] = helpful[1]/helpful[0]
except ZeroDivisionError:
  helpful['result']=0

#Filling the NaN values(created due to dividing) with 0
helpful['result'] = helpful['result'].fillna(0)

#Rounding of the results to two decimal places
helpful['result']=helpful['result'].round(2) 

#Attaching the results to a new column of the main dataframe
process_reviews['helpful_rate']=helpful['result']

#dropping the helpful column from main dataframe
process_reviews=process_reviews.drop(['helpful'], axis=1)
print(process_reviews.head())
print(process_reviews['helpful_rate'].value_counts())

#Removing unnecessary columns
process_reviews=process_reviews.drop(['reviewerName','unixReviewTime'], axis=1)
#Creating a copy 
clean_reviews=process_reviews.copy()

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

process_reviews['reviews']=process_reviews['reviews'].apply(lambda x:review_cleaning(x))
print (process_reviews.head())

stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']

process_reviews['reviews'] = process_reviews['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print (process_reviews.head())
print(pd.DataFrame(process_reviews.groupby('sentiment')['helpful_rate'].mean()))


process_reviews.groupby(['year','sentiment'])['sentiment'].count().unstack().plot(legend=True)
plt.title('Year and Sentiment count')
plt.xlabel('Year')
plt.ylabel('Sentiment count')
plt.show()

#Creating a dataframe
day=pd.DataFrame(process_reviews.groupby('day')['reviews'].count()).reset_index()
day['day']=day['day'].astype('int64')
day.sort_values(by=['day'])

# Generate a color palette with as many colors as there are days (max 31)
num_colors = day.shape[0]
colors = sns.color_palette("hsv", num_colors)

# Create barplot with custom colors
plt.figure(figsize=(12, 6))
sns.barplot(x="day", y="reviews", data=day, palette=colors)
plt.title('Day vs Reviews count')
plt.xlabel('Day')
plt.ylabel('Reviews count')
plt.show()

process_reviews['polarity'] = process_reviews['reviews'].map(lambda text: TextBlob(text).sentiment.polarity)
process_reviews['review_len'] = process_reviews['reviews'].astype(str).apply(len)
process_reviews['word_count'] = process_reviews['reviews'].apply(lambda x: len(str(x).split()))
print(process_reviews.head())

fig = px.histogram(
    process_reviews, 
    x='polarity', 
    nbins=50,
    title='Sentiment Polarity Distribution',
    labels={'polarity': 'Polarity', 'count': 'Count'}
)

fig.update_traces(marker_line_color='black')
fig.update_layout(
    xaxis_title='Polarity',
    yaxis_title='Count'
)

fig.show()


# 1.Review Rating Distribution
fig1 = px.histogram(
    process_reviews, 
    x='overall', 
    color='overall',  # Different color per rating
    category_orders={"overall": [1, 2, 3, 4, 5]},  # Ensure proper order
    color_discrete_sequence=px.colors.qualitative.Set1,  # Distinct colors
    title='Review Rating Distribution',
    labels={'overall': 'Rating'}
)

fig1.update_traces(marker_line_color='black')
fig1.update_layout(xaxis_title='Rating', yaxis_title='Count')
fig1.show()

# 2.Review Text Length Distribution
fig2 = px.histogram(
    process_reviews, 
    x='review_len', 
    nbins=100,
    title='Review Text Length Distribution',
    labels={'review_len': 'Review Length'},
)
fig2.update_traces(marker_line_color='black')
fig2.update_layout(xaxis_title='Review Length', yaxis_title='Count')
fig2.show()

# 3. Review Text Word Count Distribution
fig3 = px.histogram(
    process_reviews, 
    x='word_count', 
    nbins=100,
    title='Review Text Word Count Distribution',
    labels={'word_count': 'Word Count'},
)
fig3.update_traces(marker_line_color='black')
fig3.update_layout(xaxis_title='Word Count', yaxis_title='Count')
fig3.show()

# N gram
# Filter Data
review_pos = process_reviews[process_reviews["sentiment"] == 'Positive'].dropna()
review_neu = process_reviews[process_reviews["sentiment"] == 'Neutral'].dropna()
review_neg = process_reviews[process_reviews["sentiment"] == 'Negative'].dropna()

# N-gram Generator
def generate_ngrams(text, n_gram=1):
    tokens = [token for token in text.lower().split(" ") if token and token not in STOPWORDS]
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Bar Chart Creator
def horizontal_bar_chart(df, color):
    return go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        orientation='h',
        marker=dict(color=color),
        showlegend=False
    )

# Function to get Top Words and Create Traces 
def get_trace_from_reviews(review_data, color):
    freq_dict = defaultdict(int)
    for sent in review_data["reviews"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    fd_sorted.columns = ["word", "wordcount"]
    return horizontal_bar_chart(fd_sorted.head(25), color)

# Create Traces
trace0 = get_trace_from_reviews(review_pos, 'green')
trace1 = get_trace_from_reviews(review_neu, 'grey')
trace2 = get_trace_from_reviews(review_neg, 'red')

# Subplot Layout
fig = make_subplots(
    rows=3, cols=1,
    vertical_spacing=0.06,
    subplot_titles=[
        "Frequent Words of Positive Reviews",
        "Frequent Words of Neutral Reviews",
        "Frequent Words of Negative Reviews"
    ]
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=2, col=1)
fig.add_trace(trace2, row=3, col=1)

fig.update_layout(
    height=700,
    width=700,
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='white',
    title_text="Word Count Plots"
)

fig.show()

# Function to Build Frequency Plot Data
def get_bigram_trace(review_data, color):
    freq_dict = defaultdict(int)
    for sent in review_data["reviews"]:
        for word in generate_ngrams(sent, 2):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    fd_sorted.columns = ["word", "wordcount"]
    return horizontal_bar_chart(fd_sorted.head(25), color)

# Generate Traces
trace0 = get_bigram_trace(review_pos, 'green')
trace1 = get_bigram_trace(review_neu, 'grey')
trace2 = get_bigram_trace(review_neg, 'brown')

# Create Subplots
fig = make_subplots(
    rows=3, cols=1,
    vertical_spacing=0.06,
    subplot_titles=[
        "Bigram Plots of Positive Reviews",
        "Bigram Plots of Neutral Reviews",
        "Bigram Plots of Negative Reviews"
    ]
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=2, col=1)
fig.add_trace(trace2, row=3, col=1)

fig.update_layout(
    height=700,
    width=700,
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='white',
    title_text="Bigram Word Count Plots"
)

fig.show()


# Frequency processor
def get_trigram_trace(reviews, color):
    freq_dict = defaultdict(int)
    for sent in reviews["reviews"]:
        for word in generate_ngrams(sent, 3):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    fd_sorted.columns = ["word", "wordcount"]
    return horizontal_bar_chart(fd_sorted.head(25), color)

trace0 = get_trigram_trace(review_pos, 'green')
trace1 = get_trigram_trace(review_neu, 'grey')
trace2 = get_trigram_trace(review_neg, 'red')

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    vertical_spacing=0.06,
    subplot_titles=[
        "Trigram Plots of Positive Reviews",
        "Trigram Plots of Neutral Reviews",
        "Trigram Plots of Negative Reviews"
    ]
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=2, col=1)
fig.add_trace(trace2, row=3, col=1)


fig.update_layout(
    height=700,
    width=700,
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='white',
    title_text="Trigram Count Plots"
)
fig.show()

text = review_pos["reviews"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

text = review_neu["reviews"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)

plt.show()

text = review_neg["reviews"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = stop_words).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# Label encoding
label_encoder = preprocessing.LabelEncoder()
process_reviews['sentiment'] = label_encoder.fit_transform(process_reviews['sentiment'])

print("Unique encoded labels:", process_reviews['sentiment'].unique())
print("Value counts:\n", process_reviews['sentiment'].value_counts())

#Extract reviews column
review_features = process_reviews[['reviews']].copy().reset_index(drop=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning and lemmatization
corpus = []
for i in range(0, len(review_features)):
    review = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])  # remove non-letters
    review = review.lower()
    tokens = word_tokenize(review)  # tokenize
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]  # lemmatize
    review = ' '.join(lemmatized)
    corpus.append(review)

print("Example lemmatized review:\n", corpus[3])

tfidf_vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,2),min_df=1, max_df=1.00)
X = tfidf_vectorizer.fit_transform(corpus)
print(X.shape)

#Getting the target variable(encoded)
y=process_reviews['sentiment']

print(f'Original dataset shape : {Counter(y)}')

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f'Resampled dataset shape {Counter(y_res)}')

# Divide the dataset into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=0)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#creating the objects
logreg_cv = LogisticRegression(random_state=0)
dt_cv=DecisionTreeClassifier()
knn_cv=KNeighborsClassifier()
svc_cv=SVC()
nb_cv=BernoulliNB()
rf_cv = RandomForestClassifier(random_state=0)

cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'SVC',4:'Naive Bayes',5: 'Random Forest'}
cv_models=[logreg_cv,dt_cv,knn_cv,svc_cv,nb_cv, rf_cv]


for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X, y, cv=10, scoring ='accuracy').mean()))

# TF-IDF output is a sparse matrix, need to convert it to dense first
# Grid Search
param_grid = {'C': np.logspace(-4, 4, 50),
             'penalty':['l1', 'l2']}
clf = GridSearchCV(LogisticRegression(random_state=0, solver='liblinear', max_iter=1000 ), param_grid,cv=5, verbose=0,n_jobs=-1)
best_model = clf.fit(X_train,y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X_test,y_test))

logreg = LogisticRegression(C=10000.0, random_state=0, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion Matrix of Logistic Regression
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Negative','Neutral','Positive'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()

print("Classification Report:\n",classification_report(y_test, y_pred))

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid_rf, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Best RF Parameters:", rf_grid.best_params_)
rf = rf_grid.best_estimator_
rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print('Accuracy of random forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))

# Confusion Matrix Random Forest
cm_rf = metrics.confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, classes=['Negative','Neutral','Positive'])
plt.title('Random Forest Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

#Binarizing the target feature
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Train-Test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# One-vs-Rest Classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=10))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute TPR, FPR and AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # FIXED LINE

mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.show() 