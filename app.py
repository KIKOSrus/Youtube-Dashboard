import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

analyzer = load_vader()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return pd.Series(['positive', scores['compound']])
    elif scores['compound'] <= -0.05:
        return pd.Series(['negative', scores['compound']])
    else:
        return pd.Series(['neutral', scores['compound']])

@st.cache_data
def generate_wordcloud(comments_series):
    text = ''.join(comments_series.astype(str).tolist())

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
    ).generate(text)

    return wordcloud

@st.cache_data
def load_data():
    df_agg = pd.read_csv('data/Aggregated_Metrics_By_Video.csv').iloc[1:]
    df_agg_sub = pd.read_csv('data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('data/All_Comments_Final.csv')
    df_time = pd.read_csv('data/Video_Performance_Over_Time.csv')
    df_agg.columns = df_agg.columns.str.replace('\xad', '', regex=False)
    df_agg.columns = df_agg.columns.str.strip()
    
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='%b %d, %Y')
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(
        lambda x: datetime.strptime(x, '%H:%M:%S').time()
    )
    df_agg['Average duration sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Likes'] + df_agg['Dislikes']) / df_agg['Views']
    df_agg['Subs gained / Views'] = df_agg['Subscribers gained'] / df_agg['Views']
    df_agg = df_agg.sort_values('Video publish time', ascending=False)
    df_time['Date'] = pd.to_datetime(df_time['Date'].str.replace('Sept', 'Sep'), format='%d %b %Y')
    return df_agg, df_agg_sub, df_comments, df_time


def color_cols(val):
    try:
        if val < 0:
            return 'color: red'
        elif val > 0:
            return 'color: green'
        return ''
    except:
        return ''


df_agg, df_agg_sub, df_comments, df_time = load_data()

df_agg_copy = df_agg.copy()
metric_date_year = df_agg_copy['Video publish time'].max() - pd.DateOffset(years=1)

numeric_cols = df_agg_copy.select_dtypes(include='number').columns

median_agg = df_agg_copy[
    df_agg_copy['Video publish time'] >= metric_date_year
][numeric_cols].median()


#нормализация
df_agg_copy[numeric_cols] = (
    df_agg_copy[numeric_cols] - median_agg[numeric_cols]
) / median_agg[numeric_cols]


#for comparing videos with others
df_time_diff = pd.merge(df_time, df_agg[['Video', 'Video publish time']], left_on='External Video ID', right_on='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

#last year

date_year = df_agg['Video publish time'].max() - pd.DateOffset(years=1)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_year]

views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0, 30)]
views_cumulative = views_days[['days_published','median_views','80pct_views','20pct_views']] 
views_cumulative[['median_views','80pct_views','20pct_views']] = views_cumulative[['median_views','80pct_views','20pct_views']].cumsum()


#сайдбар
sidebar = st.sidebar.selectbox('Mean statistics or statistics for one video', ('Mean', 'One Video'))

if sidebar == 'Mean':
    st.subheader('Mean statistics')
    df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM (USD)','Average percentage viewed (%)',
                             'Average duration sec', 'Engagement ratio','Subs gained / Views']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =12)

    df_6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo]
    df_12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo]
    numeric_cols = df_6mo.select_dtypes(include='number').columns

    metric_medians6mo = df_6mo[numeric_cols].median()
    metric_medians12mo = df_12mo[numeric_cols].median()
    st.metric('Views', metric_medians6mo['Views'], 500)

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    size = len(columns)
    c = 0
    for i in metric_medians6mo.index:
        with columns[c]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label= i, value = round(metric_medians6mo[i],3), delta = "{:.2%}".format(delta))
            c += 1
            if c >= size:
                c = 0
    
    df_agg_copy['Publish Date'] = df_agg_copy['Video publish time'].apply(lambda x: x.date())
    df_agg_final = df_agg_copy[['Video title', 'Publish Date', 'Views', 'Likes', 'Subs gained / Views', 'Engagement ratio']]

    numeric_cols = df_agg_final.select_dtypes(include=[np.number]).columns

    format_dict = {col: "{:.1%}" for col in numeric_cols}
    
    n_recent = 5
    st.subheader(f"{n_recent} Recent Videos")
    st.dataframe(df_agg_final.head(n_recent).style.map(color_cols).format(format_dict))
    st.subheader("Top 3 best and worst videos")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top 3 best")
        top_views = df_agg.nlargest(3, 'Views')[['Video title', 'Views']]
        st.dataframe(top_views)
    with col2:
        st.write("Top 3 worst")
        bottom_views = df_agg.nsmallest(3, 'Views')[['Video title', 'Views']]
        st.dataframe(bottom_views)

elif sidebar == 'One Video':
    st.subheader('Individual Video')
    video_titles = df_agg['Video title']

    video_selected = st.selectbox("Choose the video", 
                 options=video_titles)

    df_agg_filtered = df_agg[df_agg['Video title'] == video_selected]
    df_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_selected]
    df_sub_filtered = df_sub_filtered.sort_values('Is Subscribed')

    fig = px.bar(df_sub_filtered, x='Views', y='Is Subscribed', color='Country Code', orientation='h')
    st.plotly_chart(fig)

    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_selected]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0, 30)]
    first_30 = first_30.sort_values('days_published')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                    mode='lines',
                    name='20th percentile', line=dict(color='purple', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                        mode='lines',
                        name='50th percentile', line=dict(color='black', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                        mode='lines', 
                        name='80th percentile', line=dict(color='royalblue', dash ='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                        mode='lines', 
                        name='Current Video' ,line=dict(color='firebrick',width=8)))
        
    fig2.update_layout(title='View comparison first 30 days',
                   xaxis_title='Days Since Published',
                   yaxis_title='Cumulative views')

    st.plotly_chart(fig2)
    st.write('Comments analysis')
    
    #id video
    video_id = df_agg_filtered['Video'].iloc[0]
    comments_series = df_comments[df_comments['VidId'] == video_id]['Comments']
    result_df = comments_series.apply(get_sentiment)
    result_df.columns = ['sentiment', 'compound']

    sentiment_analysis = pd.DataFrame({
        'comment': comments_series,
        'sentiment': result_df['sentiment'],
        'compound': result_df['compound']
    })
    st.dataframe(sentiment_analysis)

    # wordcloud
    st.subheader('Wordcloud')
    wordcloud = generate_wordcloud(comments_series)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)
