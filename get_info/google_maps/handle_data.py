import pandas as pd
from datetime import datetime, timedelta


time_units = {'день': timedelta(days=1), 'дн': timedelta(days=1), 'недел': timedelta(weeks=1)}
now = datetime.now()

def text_to_date(text):
    global time_units, now

    date = text.rsplit(' ', 1)[0]
    
    if date[0].isdigit():
        num, unit = date.split(' ')
        num = int(num)
    else:
        unit = date
        num = 1
    
    if unit.startswith('год') or unit == 'лет':
        # date = datetime(now.year - num, now.month, now.day, now.hour, now.minute, now.microsecond)
        date = now.replace(year=now.year - num)
    elif unit.startswith('месяц'):

        date = now.replace(year = now.year if now.month > num else now.year - 1, 
                           month = now.month - num if now.month > num else  now.month - num + 12)
    else:
        for key in time_units:
            if unit.startswith(key):
                unit = key
                break

        date = now - num*time_units[unit]

    return date.timestamp()


df = pd.read_csv('google_reviews1.csv')
df['rating'] = df['rating'].apply(lambda x: int(x.split(' ')[0]))
df['date'] = df['date'].str.lower().apply(text_to_date)
df['review'] = df['review'].str.replace('\n', ' ').str.replace('\t', ' ')
print(df)
