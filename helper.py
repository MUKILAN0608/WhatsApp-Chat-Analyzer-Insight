from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
# Initialize URL extractor
extractor = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    total_messages = df.shape[0]
    total_words = df['message'].apply(lambda x: len(x.split())).sum()
    media_messages = df[df['message'] == "<Media omitted>"].shape[0]

    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    total_links = len(links)

    return total_messages, total_words, media_messages, total_links


def get_user_list(df):
    user_list = df['user'].unique().tolist()
    if 'notification' in user_list:
        user_list.remove('notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    return user_list


def filter_by_user(selected_user, df):
    return df if selected_user == "Overall" else df[df['user'] == selected_user]


def most_busy_users(df):
    message_count = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    df_percent.columns = ['name', 'percent']
    return message_count, df_percent


def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(df['message'].str.cat(sep=" "))


def most_common_words(selected_user, df, stop_words):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'notification']
    temp = temp[temp['message'] != '<Media omitted>']

    words = []
    for message in temp['message']:
        message = message.lower()
        for word in message.split():
            if word not in stop_words and word.strip() != '':
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


def emoji_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])


def monthly_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'datetime' not in df.columns:
        raise KeyError("'datetime' column is missing. Check preprocessing step.")

    # Convert to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    df['month'] = df['datetime'].dt.strftime('%b')    # e.g., Jan, Feb
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month

    # Group by year and month number
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    # Sort by year and month number
    timeline = timeline.sort_values(by=['year', 'month_num'])

    # Combine year and month for timeline axis
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)

    # Rename message count column to messages
    timeline.rename(columns={'message': 'messages'}, inplace=True)

    # Final structure expected by app
    timeline = timeline[['time', 'messages']]
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'datetime' not in df.columns:
        raise KeyError("'datetime' column is missing. Check preprocessing step.")

    # Extract only date part (no time)
    df['only_date'] = df['datetime'].dt.date

    # Count messages per day
    daily_timeline = df.groupby('only_date').count()['message'].reset_index(name='message_count')
    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if df.empty:
        return pd.Series(dtype='int')
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if df.empty:
        return pd.Series(dtype='int')
    return df['month'].value_counts()

def activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    activity_heatmap=df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count')
    return activity_heatmap



def calculate_response_times(df):
    df = df[df['user'] != 'notification'].copy()
    df = df.sort_values('datetime')
    df = df.reset_index(drop=True)

    df['next_user'] = df['user'].shift(-1)
    df['next_time'] = df['datetime'].shift(-1)

    df['is_reply'] = df['user'] != df['next_user']
    reply_df = df[df['is_reply']].copy()

    reply_df['response_time'] = (reply_df['next_time'] - reply_df['datetime']).dt.total_seconds() / 60
    reply_df = reply_df[reply_df['response_time'] <= 720]  # Discard weird long delays

    reply_df['responder'] = reply_df['next_user']
    reply_df['to'] = reply_df['user']

    pairwise_avg = reply_df.groupby(['responder', 'to'])['response_time'].mean().reset_index()
    pairwise_avg.columns = ['Responder', 'To', 'AvgResponseTime']
    return pairwise_avg


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))






