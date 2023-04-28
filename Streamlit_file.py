from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2074&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
add_bg_from_url()
st.header('Flysafe Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input("Text here:")
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))
    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True,
                                 punct=True))
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')


    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    #
    def analyze(x):
        if x >= 0.1:
            return 'Positive'
        elif x <= 0:
            return 'Negative'
        else:
            return 'Neutral'


    #
    if upl:
        df = pd.read_csv(upl)
        df['score'] = df['comment'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        df['subjectivity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        st.write(df.head(10))


        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')


        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',)
        df['polarity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        # Show Statisitcs
        st.write('**Statistics:**')
        st.write(df.describe())
        # Show pie chart
        st.write('**Pie Chart of Analysis Report:**')
        fig, ax = plt.subplots()
        sentiment_counts = df['analysis'].value_counts()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
        # Show Distribution
        st.write('**Distribution:**')
        column = st.selectbox('Select a column', ['polarity', 'subjectivity'])
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
        st.write('**Distribution:**')
        column = st.selectbox('Select a column', ['score','analysis'])
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
        # Show scatter plot
        st.write('**Scatter Plot:**')
        x = st.selectbox('Select X-axis', ['analysis'])
        y = st.selectbox('Select Y-axis', ['polarity', 'subjectivity'])
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)
        # Show Correlation
        st.write('**Correlation:**')
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        # Show histogram of sentiment polarity
        st.write('**Histogram of Sentiment Polarity:**')
        fig, ax = plt.subplots()
        sns.histplot(df['analysis'], kde=True, ax=ax)
        st.pyplot(fig)
        # Generate word cloud
        st.write('**Word Cloud:**')
        fig, ax = plt.subplots()
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(df['comment']))
        ax.imshow(wordcloud)
        ax.axis('off')
        st.pyplot(fig)
        # Generate negative word cloud
        st.write('**Negative Word Cloud:**')
        fig, ax = plt.subplots()
        wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='Reds', max_words=50,
                              ).generate(
            ' '.join(df[df['polarity'] < 0]['comment']))
        ax.imshow(wordcloud)
        ax.axis('off')
        st.pyplot(fig)






