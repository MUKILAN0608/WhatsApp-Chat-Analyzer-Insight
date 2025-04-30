import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import plotly.express as px
import nltk
from nltk.corpus import stopwords
import networkx as nx
import plotly.graph_objects as go

# Custom imports (make sure these files are in your working directory)
import preprocessor
import helper
import sentiment_analysis
import interaction_graph

# Set Streamlit page configuration
st.set_page_config(page_title="📊 WhatsApp Chat Analyzer", layout="wide")

# Load emoji-supporting font
emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"
emoji_font = None
try:
    emoji_font = fm.FontProperties(fname=emoji_font_path)
except Exception:
    st.warning("Emoji-support font not found. Emojis may not render properly.")

emoji_font_kwargs = {'fontproperties': emoji_font} if emoji_font else {}

# Sidebar for file upload
st.sidebar.title("📂 Upload Chat File")
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp .txt file", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = helper.get_user_list(df)
    selected_user = st.sidebar.selectbox("Select user for analysis", user_list)
    analyze = st.sidebar.button("🔍 Run Analysis")

    if analyze:
        st.title("📋 Full Chat Data")
        st.dataframe(df, use_container_width=True)

        st.markdown(f"## 📁 Chat Overview: **{selected_user}**")

        # Load stopwords
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                stop_words = f.read().splitlines()
        except FileNotFoundError:
            st.error("❌ 'stopwords.txt' not found.")
            stop_words = []

        filtered_df = helper.filter_by_user(selected_user, df)
        total_messages, total_words, media_messages, total_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📨 Messages", total_messages)
        col2.metric("✍️ Words", total_words)
        col3.metric("🖼️ Media", media_messages)
        col4.metric("🔗 Links", total_links)

        if selected_user == 'Overall':
            st.subheader("📈 Most Active Participants")
            message_count, df_percent = helper.most_busy_users(df)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=message_count.index, y=message_count.values, ax=ax, palette="mako")
            ax.set_title("Messages by User", fontsize=14)
            ax.set_ylabel("Messages")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            st.dataframe(df_percent, use_container_width=True)

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(df_wc)
        ax.axis("off")
        st.pyplot(fig)

        # Common Words
        st.subheader("🐠 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df, stop_words)
        if not most_common_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(x='count', y='word', data=most_common_df, palette="viridis", ax=ax)
            ax.set_title("Top 20 Words")
            st.pyplot(fig)
        else:
            st.info("No common words found.")

        # Emoji analysis
        st.subheader("😊 Emoji Usage")
        emoji_df = helper.emoji_analysis(selected_user, df)
        if not emoji_df.empty:
            top_emojis = emoji_df.head(10)
            fig, ax = plt.subplots(figsize=(7, 7))
            wedges, texts, autotexts = ax.pie(
                top_emojis['count'],
                labels=top_emojis['emoji'],
                autopct='%1.1f%%',
                startangle=140,
                textprops={**{'fontsize': 13}, **emoji_font_kwargs},
                wedgeprops=dict(edgecolor='white', linewidth=1.5)
            )
            ax.set_title("Top Emoji Usage", **emoji_font_kwargs)
            st.pyplot(fig)
            with st.expander("📋 Full Emoji Frequency"):
                st.dataframe(emoji_df, use_container_width=True)
        else:
            st.info("No emojis used.")

        # Monthly Timeline
        st.subheader("🗓️ Monthly Timeline")
        timeline = helper.monthly_analysis(selected_user, df)
        if not timeline.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            y_col = next((col for col in ['message', 'messages', 'count'] if col in timeline.columns), None)
            if y_col:
                ax.plot(timeline['time'], timeline[y_col], marker='o', color='teal')
                ax.set_xlabel("Month-Year")
                ax.set_ylabel("Messages")
                ax.set_title("Monthly Activity", fontsize=14)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
            else:
                st.error("Expected message count column not found.")
                st.dataframe(timeline)
        else:
            st.info("Not enough data for timeline.")

        # Daily Timeline
        st.subheader("🗓️ Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        if not daily_timeline.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(daily_timeline['only_date'], daily_timeline['message_count'], marker='o', color='orangered')
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            ax.set_title("Daily Activity Timeline", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Not enough data for daily timeline.")

        # Activity Maps
        st.markdown("---")
        st.title("🎛️📈 Activity Map Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='#72B7B2')
            ax.set_xlabel("Day")
            ax.set_ylabel("Message Count")
            ax.set_title("Most Busy Day")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='#F18F01')
            ax.set_xlabel("Month")
            ax.set_ylabel("Message Count")
            ax.set_title("Most Busy Month")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Weekly Activity Heatmap
        st.title("🗺️ Weekly Activity Heatmap")
        user_heatmap = helper.activity_map(selected_user, df)
        if not user_heatmap.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(user_heatmap, ax=ax, cmap="rocket", annot=True)
            ax.set_title("Activity Heatmap (Weekday vs Hour)")
            st.pyplot(fig)
        else:
            st.info("Not enough data for heatmap.")

        # Average Response Time
        st.header("🕒 Average Response Time Analysis")
        with st.spinner("Calculating..."):
            pairwise_avg = helper.calculate_response_times(df)
            if selected_user != "Overall":
                filtered = pairwise_avg[pairwise_avg['Responder'] == selected_user]
                if not filtered.empty:
                    avg_time = round(filtered['AvgResponseTime'].mean(), 2)
                    st.success(f"💬 {selected_user}'s average reply time: **{avg_time} minutes**")
                else:
                    st.info(f"No replies from {selected_user} to analyze.")
            else:
                st.success("Showing group-wide response patterns.")

        st.subheader("📊 Who replies fastest to whom?")
        pivot_df = pairwise_avg.pivot(index='Responder', columns='To', values='AvgResponseTime')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, cbar_kws={"label": "Minutes"})
        plt.title("Average Response Time (min)")
        st.pyplot(fig)

        # Sentiment Analysis
        nltk.download('stopwords')
        stop_words_nltk = list(set(stopwords.words('english')))
        df_with_sentiment, sentiment_summary = sentiment_analysis.analyze_sentiment(df, selected_user)

        st.subheader("🧐 Sentiment Analysis")
        fig = px.pie(
            sentiment_summary,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution",
            color_discrete_sequence=["#66c2a5", "#fc8d62", "#8da0cb"]
        )
        st.plotly_chart(fig, use_container_width=True)

        total = sentiment_summary['Count'].sum()
        col1, col2, col3 = st.columns(3)
        for i, row in sentiment_summary.iterrows():
            sentiment = row['Sentiment']
            count = row['Count']
            percent = (count / total) * 100

            if sentiment == "POSITIVE":
                icon, color = "😊", "green"
            elif sentiment == "NEGATIVE":
                icon, color = "😠", "red"
            else:
                icon, color = "😐", "gray"

            with [col1, col2, col3][i]:
                st.markdown(f"""
                    <div style="padding: 10px; background-color: {color}; border-radius: 10px; color: white; text-align: center;">
                        <h3 style="margin: 0;">{icon}</h3>
                        <p style="margin: 5px 0;">{sentiment}</p>
                        <p style="margin: 0;"><b>{count}</b> messages</p>
                        <small>{percent:.1f}%</small>
                    </div>
                """, unsafe_allow_html=True)

        with st.expander("📂 View Sentiment-Labeled Messages"):
            st.dataframe(df_with_sentiment[['datetime', 'user', 'message', 'sentiment']].sort_values('datetime'), use_container_width=True)

        # Interaction Network Graph
        st.subheader("🧑‍🤝‍🧑 User Interaction Network")

        try:
            G, node_stats, fig = interaction_graph.get_interaction_graph(df)
            node_colors = [node_stats[node] for node in G.nodes()]
            node_sizes = [node_stats[node] * 3 for node in G.nodes()]

            pos = nx.spring_layout(G)
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            edge_x, edge_y = [], []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color='gray'),
                hoverinfo='none',
                mode='lines'
            )

            node_labels = [f'{node}<br>Messages: {node_stats[node]}' for node in G.nodes()]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=node_labels,
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    color=node_colors,
                    size=node_sizes,
                    colorbar=dict(
                        thickness=15,
                        title=dict(
                            text='Message Volume',
                            side='right'
                        ),
                        xanchor='left'
                    ),
                    line_width=2
                ),
                textposition='bottom center'
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=dict(
                                    text='User Interaction Graph',
                                    font=dict(
                                        size=16
                                    )
                                ),
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[dict(
                                    text="Message transitions between users",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002)],
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False)
                            ))

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to generate interaction graph: {e}")

else:
    st.warning("📁 Please upload a WhatsApp chat file (.txt) to begin analysis.")