from flask import Flask, render_template, request
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data
df = pd.read_excel('laws.xlsx')


# Preprocess Data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

df['processed_description'] = df['Description'].apply(preprocess_text)

@app.route('/')
def index():
    return render_template('/lawai.io/index.html')

@app.route('/lawai.io/result', methods=['POST'])
def result():
    user_input = request.form['incident']
    processed_user_input = preprocess_text(user_input)

    all_descriptions = df['processed_description'].tolist()
    all_descriptions.append(processed_user_input)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_descriptions)

    tfidf_matrix_train = tfidf_matrix[:-1].toarray()
    tfidf_matrix_user = tfidf_matrix[-1].toarray()

    cosine_similarities = cosine_similarity(tfidf_matrix_train, tfidf_matrix_user)
    most_similar_index = cosine_similarities.argmax()

    relevant_ipc_section = df.iloc[most_similar_index]['IPC Section']
    relevant_description = df.iloc[most_similar_index]['Description']

    return render_template('/result.html', ipc_section=relevant_ipc_section, description=relevant_description)

if __name__ == '__main__':
    app.run(debug=True)
