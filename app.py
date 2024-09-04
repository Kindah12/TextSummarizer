from flask import Flask, render_template, request
import math
import nltk
from nltk.corpus import stopwords


# A function to prepare the text for summarization by tokenizing sentences into words, 
# removing stop words, and counting the frequency of remaining words in each sentence.
def word_freq(sentences):

    freq_matrix = {}

    for sentence in sentences:
        freq_words={}
        words = nltk.word_tokenize(sentence)
        stop_words = set(stopwords.words('english'))
        for word in words:
            if word.lower() not in stop_words:
                if word in freq_words:
                    freq_words[word] += 1
                else:
                    freq_words[word] = 1
        freq_matrix[sentence] = freq_words

    return freq_matrix


# A function to calculate the term frequency (TF) of each word in each sentence,
# which is computed as the number of occurrences of each word divided by
# the total number of words in the sentence.
def term_frequency(freq_matrix):

    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sent] = tf_table

    return tf_matrix


# A function to calculate the inverse document frequency (IDF) for each word in sentence,
# which is computed as the logarithm of the total number of sentences divided by 
# the occurrences of the word on the sentences. 
def idf(sentences_count,freq_matrix):

    idf_matrix1 = {}
    count_words = {}

    for sentence,freq_word in freq_matrix.items():
        for word in freq_word:
            if word not in count_words:
                count_words[word] = 1
            else:
                count_words[word] += 1

    for sentence, freq_word in freq_matrix.items():
        x = {}
        for word in freq_word:
            if word not in x:
                x[word] = math.log10(sentences_count / float(count_words[word]))
            else:
                continue
        idf_matrix1[sentence] = x

    return idf_matrix1

# A function to calculate the TF-IDF score for each word in each sentence.
# which is computed by multiplying the term frequency (TF) of each word 
# with its inverse document frequency (IDF). 
def tf_idf(tf_matrix,idf_matrix):

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),f_table2.items()): 
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

# A function to calculate a score for each sentence based on its TF-IDF values,
# The score is computed as the average of the TF-IDF scores of the words in the sentence.
def sentences_score(tf_idf_matrix):

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

# A function to generate a summary of the input text using TF-IDF scores.
def generate_summary(text):

    sentences = nltk.sent_tokenize(text)
    words_mtrx = word_freq(sentences)
    tf_matrix = term_frequency(words_mtrx)
    sentences_count = len(sentences)
    idf_matrix = idf(sentences_count,words_mtrx)
    tf_idf_matrix = tf_idf(tf_matrix,idf_matrix)
    sentenceValue = sentences_score(tf_idf_matrix)
    sumValues = 0
    summary = ''

    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    for sentence,value in sentenceValue.items():
        if value >= (average):
            summary += " " + sentence

    return summary


app = Flask(__name__)
@app.route('/',methods=["POST", "GET"])
def home():
    if request.method == "POST" and request.form["text"] != '':
        text = request.form["text"]
        result = generate_summary(text)
    else:
        text = ''
        result = ''
    return render_template("homepage.html", result=result, text=text)


if __name__ == '__main__':
    app.run(debug=True)