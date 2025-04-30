import re
import pandas as pd

def preprocess(data):
    # Regex pattern for WhatsApp messages
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})\s?(am|pm|AM|PM|[ap]m)?\s-\s(.*)'
    matches = re.findall(pattern, data, re.MULTILINE)

    dates = []
    messages = []

    for match in matches:
        time_part = f"{match[1]} {match[2]}" if match[2] else match[1]
        date_str = f"{match[0]}, {time_part}"
        dates.append(date_str)
        messages.append(match[3])

    df = pd.DataFrame({'user_message': messages, 'date': dates})

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y, %I:%M %p', errors='coerce')
    df = df.dropna(subset=['date']).reset_index(drop=True)

    # Rename for consistency
    df.rename(columns={'date': 'datetime'}, inplace=True)

    # Split user from message
    users = []
    messages_list = []
    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1].strip())
            messages_list.append(entry[2].strip())
        else:
            users.append('notification')
            messages_list.append(entry[0].strip())

    df['user'] = users
    df['message'] = messages_list
    df.drop(columns=['user_message'], inplace=True)

    # Time-based columns
    df['only_date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_name'] = df['datetime'].dt.day_name()

    def get_period(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'  # For hours between 12 AM - 6 AM

    df['period'] = df['hour'].apply(get_period)

    return df

