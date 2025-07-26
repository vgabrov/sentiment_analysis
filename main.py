#!/usr/bin/env python
# coding: utf-8

#!pip install selenium transformers torch numpy pandas tqdm seaborn matplotlib statsmodels wordcloud nltk feedparser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from wordcloud import WordCloud
#from nltk.corpus import stopwords
#import nltk

# ----- Парсинг отзывов с Яндекс.Карт -----

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

user_input = input("Введите ссылку на страницу отзывов объекта Яндекс.Карт (или нажмите Enter для использования по умолчанию): ").strip()
url = user_input if user_input else "https://yandex.ru/maps/org/sankt_peterburgskiy_gosudarstvenny_universitet/1040743234/reviews/"
driver.get(url)

total_reviews_text = driver.find_element(By.XPATH, '//*[@class="card-section-header__title _wide"]').text
total_reviews_int = int(re.sub(r'\D', '', total_reviews_text))

reviews_elements = set()
pbar = tqdm(total=total_reviews_int)
pbar.set_description("Loading all reviews on the page")

while total_reviews_int != len(reviews_elements):
    old_len = len(reviews_elements)
    for review_elem in driver.find_elements(By.XPATH, '//*[@class="business-review-view__info"]'):
        reviews_elements.add(review_elem)
        driver.execute_script("arguments[0].scrollIntoView(true);", review_elem)
    pbar.update(len(reviews_elements) - old_len)
    time.sleep(0.3)

pbar.close()
driver.quit()

reviews = []
dates = []

for el in reviews_elements:
    try:
        text = el.find_element(By.CLASS_NAME, "spoiler-view__text-container").text
        reviews.append(text)
        meta = el.find_element(By.XPATH, './/meta[@itemprop="datePublished"]')
        dates.append(meta.get_attribute("content"))
    except Exception:
        continue

df = pd.DataFrame({
    "comment": reviews,
    "date": pd.to_datetime(dates)
})

# ----- Обработка текста -----

df["comment"] = df["comment"].str.lower().str.replace(r"[^\w\s!?]", "", regex=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ----- Оценка сентиментов моделью rubert -----

model_name = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

batch_size=32
comments=df["comment"].tolist()

results=[]

for i in range (0,len(comments),batch_size):
    batch = comments[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        results.append(probs)

cols_to_drop = [col for col in df.columns if col.startswith("rubert_")]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

all_probs = np.vstack(results)

df["rubert_neutral"] = all_probs[:, 0]
df["rubert_positive"] = all_probs[:, 1]
df["rubert_negative"] = all_probs[:, 2]

# ----- Оценка с сарказмом, агрессией и тревожностью -----

model_name = "Kostya165/rubert_tiny2_russian_emotion_sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

probs_all = []

with torch.no_grad():
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i+batch_size]
        inputs = tokenizer(batch_comments, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_all.append(probs.cpu().numpy())

probs_all = np.concatenate(probs_all, axis=0)

df["rubert_kostya_aggression"] = probs_all[:, 0]
df["rubert_kostya_anxiety"] = probs_all[:, 1]
df["rubert_kostya_neutral"] = probs_all[:, 2]
df["rubert_kostya_positive"] = probs_all[:, 3]
df["rubert_kostya_sarcasm"] = probs_all[:, 4]

# ----- Анализ по неделям -----

df["week"] = df["date"].dt.isocalendar().week

weekly_stats = df.groupby("week")[[
    "rubert_negative", "rubert_neutral", "rubert_positive",
    "rubert_kostya_aggression", "rubert_kostya_anxiety",
    "rubert_kostya_sarcasm", "rubert_kostya_neutral", "rubert_kostya_positive"
]].mean()

weekly_stats_diff = weekly_stats.diff().dropna()
weekly_stats_pct = weekly_stats_diff / weekly_stats.shift(1).loc[weekly_stats_diff.index]

for col in weekly_stats_diff.columns:
    print(f"--- {col} ---")
    for idx, val in weekly_stats_diff[col].items():
        means_1 = weekly_stats.at[idx,col]
        means_0 = weekly_stats.at[idx-1,col] if idx-1 in weekly_stats.index else None
        print(f"средние за недели {idx-1}, {idx}: {means_0}, {means_1}")
        print(f"разница средних за 2 недели: {val}")
        pct = weekly_stats_pct.at[idx, col]
        print(f"% изменение: {pct:.2%}")
    print()

# ----- t-тест между неделями -----

significant_results = []

for column_name in weekly_stats.columns:
    print(f"t-тест для {column_name}")
    for i in range(1, len(weekly_stats)):
        week_prev = weekly_stats.index[i - 1]
        week_curr = weekly_stats.index[i]

        week_prev_values = df[df["week"] == week_prev][column_name]
        week_curr_values = df[df["week"] == week_curr][column_name]

        if len(week_prev_values) < 2 or len(week_curr_values) < 2:
            continue
        if np.var(week_prev_values) == 0 or np.var(week_curr_values) == 0:
            continue

        t_stat, p_value = stats.ttest_ind(week_prev_values, week_curr_values, equal_var=False)
        print(f"Сравнение недели {week_prev} и {week_curr}: t = {t_stat:.3f}, p = {p_value:.3f}")
        if p_value < 0.05:
            significant_results.append((column_name, week_prev, week_curr, t_stat, p_value))
    print()

print("Значимые изменения (p < 0.05):")
for col, w_prev, w_curr, t, p in significant_results:
    print(f"{col}: неделя {w_prev} vs {w_curr} — t={t:.3f}, p={p:.3f}")

# ----- Визуализация и графики -----

# (твой код по seaborn, matplotlib, wordcloud, регрессия и прочее оставь без изменений,
# просто после этого блока анализа)
# # Картинки

# In[18]:


#!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

df["week"] = df["week"].astype(int)

# Подготовим датафрейм в "длинный" формат (long)
df_long = pd.melt(
    df,
    id_vars=['week'],
    value_vars=['rubert_negative', 'rubert_neutral', 'rubert_positive', 
                'rubert_kostya_aggression', 'rubert_kostya_anxiety', 
                'rubert_kostya_sarcasm', 'rubert_kostya_neutral', 'rubert_kostya_positive'],
    var_name='sentiment',
    value_name='score'
)

plt.figure(figsize=(12, 6))
sns.stripplot(data=df_long, x='sentiment', y='score', hue='week', dodge=True, alpha=0.5)
plt.title('Распределение значений сентиментов по неделям')
plt.xticks(rotation=45)
plt.legend(title='Week')
plt.show()


# In[19]:


import seaborn as sns

daily_stats = df.groupby("date")[["rubert_negative", "rubert_neutral", "rubert_positive",
                                  "rubert_kostya_aggression", "rubert_kostya_anxiety",
                                  "rubert_kostya_sarcasm", "rubert_kostya_neutral", "rubert_kostya_positive"]].mean()

corr_matrix = daily_stats.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask = mask)
plt.title("Корреляция сентиментов по дням")
plt.show()


# In[20]:


import math

sentiments = ["rubert_negative", "rubert_neutral", "rubert_positive",
              "rubert_kostya_aggression", "rubert_kostya_anxiety",
              "rubert_kostya_sarcasm", "rubert_kostya_neutral", "rubert_kostya_positive"]

n_parts = 4
weeks = sorted(df["week"].unique())
# Число недель на части
chunk_size = math.ceil(len(weeks) / n_parts)

# Создаем подряд идущие группы
weeks_split = [weeks[i*chunk_size:(i+1)*chunk_size] for i in range(n_parts)]


for part_idx, weeks_subset in enumerate(weeks_split):
    fig, axes = plt.subplots(len(sentiments), 1, figsize=(12, 4 * len(sentiments)), sharex=True)

    for i, sentiment in enumerate(sentiments):
        sns.violinplot(
            x="week", y=sentiment, data=df[df["week"].isin(weeks_subset)],
            ax=axes[i], order=weeks_subset, palette="Set2", hue="week", legend=False
        )
        axes[i].set_title(f"{sentiment} — Недели: {min(weeks_subset)}–{max(weeks_subset)}")
        axes[i].set_ylabel("Значение")
        axes[i].grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        axes[i].set_xlabel("")

        sns.boxplot(
            data=df[df["week"].isin(weeks_subset)],
            x="week", y=sentiment,
            ax=axes[i],
            width=0.1,
            showcaps=False,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.2),
            medianprops=dict(color="red", linewidth=2),
            showfliers=False
        )

    axes[-1].set_xticks(range(len(weeks_subset)))
    axes[-1].set_xticklabels([f"Нед. {w}" for w in weeks_subset], rotation=45)
    plt.tight_layout()
    plt.show()


# In[21]:


plt.figure(figsize=(14, 6))

# Фон с резкими полосами для недель
for i, week in enumerate(weeks):
    plt.axvspan(i - 0.5, i + 0.5, color=('white' if i % 2 == 0 else '#f0f0f0'), zorder=0)

# Линии по каждому сентименту
for sentiment in sentiments:
    weekly_avg = df.groupby("week")[sentiment].mean().reindex(weeks)
    plt.plot(range(len(weeks)), weekly_avg, label=sentiment)
    if not weekly_avg.isna().all():
        plt.text(
            x=len(weeks) - 1 ,
            y=weekly_avg.iloc[-1],
            s=sentiment,
            fontsize=8,
            va='center'
        )

# Настройка осей
# Подпись в конце линии (последняя точка)

plt.xticks(ticks=range(len(weeks)), labels=[f"Неделя {w}" for w in weeks])
plt.xlabel("Неделя")
plt.ylabel("Среднее значение сентимента")
plt.title("Изменение сентиментов по неделям")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.xlim(-0.1, len(weeks) - 0.8)
plt.show()


# In[23]:


#!pip install statsmodels
import statsmodels.api as sm

for sentiment in sentiments:
    X = df['week']
    y = df[sentiment]
    X = sm.add_constant(X)  # добавляем свободный член
    
    model = sm.OLS(y, X).fit()
    print(f"Регрессия для {sentiment}:")
    print(model.summary())  # здесь вся статистика

    # Визуализация
    plt.scatter(df['week'], y, alpha=0.3)
    plt.plot(df['week'], model.predict(X), color='red')
    plt.title(f"Тренд по неделям: {sentiment}")
    plt.xlabel("Неделя")
    plt.ylabel("Значение сентимента")
    plt.show()


# In[28]:


#!pip install wordcloud
#!pip install nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#nltk.download('stopwords')

#russian_stopwords = set(stopwords.words('russian'))
russian_stopwords =set()

# Можно дополнительно добавить свои слова
custom_stopwords = {'это', 'как', 'вот', 'так', 'же', 'ну', 'ли', 'бы', 'только', 'не', 'все', 'на', 'и', 'о', 'об', 'в', 'за', 'по', 'с', 'у', 'но', 'нет', 'без', 'всё', 'к', 'а', 'ещё', 'был', 'была', 'было', 'были', 'что', 'из', 'я', 'ты', 'мы', 'до', 'от', 'на', '', '', '' }
russian_stopwords.update(custom_stopwords)

# Объединяем все тексты отзывов в одну строку
text = " ".join(df["comment"].dropna().tolist()).lower()
filtered_words = [word for word in text.split() if word not in russian_stopwords]
filtered_text = " ".join(filtered_words)

# Создаем облако слов
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    max_words=200,
    collocations=False,
    regexp=r"\b\w+\b"  # Чтобы включать только слова
).generate(filtered_text)

# Отобразить облако слов
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


