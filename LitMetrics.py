import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
import os

plt.rcParams['font.family'] = 'DejaVu Sans'

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Предупреждение при загрузке NLTK: {e}")

class LiteraryAnalyzer:
    def __init__(self):
        """Инициализация анализатора"""
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        self.stop_words.update(['это', 'весь', 'свой', 'мочь', 'ещё', 'также', 
                                'который', 'очень', 'самый', 'такой', 'даже'])
        
        self.word_corrections = {} 
        
    def fetch_text(self, url):
        """Получение текста произведения с сайта"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            response.raise_for_status()
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                          'aside', 'iframe', 'noscript', 'img', 'svg']):
                element.decompose()
            
            for element in soup.find_all(['ul', 'ol']):
                if element.get('class'):
                    classes = ' '.join(element.get('class'))
                    if any(x in classes.lower() for x in ['menu', 'nav', 'link']):
                        element.decompose()
            
            for selector in ['.site-header', '.site-nav', '.breadcrumbs', '.comments', 
                           '.share', '.sidebar', '.footer', '.widget', '.related',
                           'form', 'button', '#toc', '.toc', '.author-info']:
                for element in soup.select(selector):
                    element.decompose()
            
            text_content = None
            
            if 'azbyka.ru' in url:
                text_content = soup.find('div', class_='article__text')
                if not text_content:
                    text_content = soup.find('div', class_='entry-content')
                if not text_content:
                    text_content = soup.find('article')
            else:
                text_content = soup.find('article')
                
            if text_content:
                paragraphs = text_content.find_all(['p', 'div'])
                text_parts = []
                
                for p in paragraphs:
                    if p.get('class'):
                        classes = ' '.join(p.get('class'))
                        if any(skip in classes.lower() for skip in ['nav', 'menu', 'button', 
                                                                    'link', 'breadcrumb', 
                                                                    'share', 'related']):
                            continue
                    
                    text = p.get_text(separator=' ', strip=True)
                    if text and len(text) > 30:
                        if not any(x in text.lower() for x in ['см. тест', 'радиоспектакль', 
                                                               'прослушать', 'голоса из']):
                            text_parts.append(text)
                
                full_text = ' '.join(text_parts)
                return full_text
            return None
            
        except requests.exceptions.RequestException as e: 
            print(f"Сетевая ошибка при загрузке {url}: {e}")
            return None
        except Exception as e:
            print(f"Непредвиденная ошибка при парсинге {url}: {e}")
            return None
    
    def preprocess_text(self, text):
        """Предобработка текста: токенизация, лемматизация, удаление стоп-слов"""
        if not text:
            return []
        
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        try:
            tokens = word_tokenize(text, language='russian')
        except Exception as e:
            print(f"Ошибка токенизации, используем базовый метод: {e}")
            tokens = text.split()
        
        lemmas = []
        for token in tokens:
            if len(token) > 2 and token.isalpha() and token not in self.stop_words:
                try:
                    parsed = self.morph.parse(token)[0]
                    lemma = parsed.normal_form
                    
                    if lemma in self.word_corrections:
                        lemma = self.word_corrections[lemma]

                    if parsed.tag.POS not in ['PREP', 'CONJ', 'PRCL', 'INTJ']:
                        lemmas.append(lemma)
                except Exception:
                    continue
        
        return lemmas
    
    def get_word_frequency(self, lemmas, top_n=50):
        """Подсчет частоты слов"""
        return Counter(lemmas).most_common(top_n)
    
    def create_wordcloud(self, lemmas, title):
        """Создание облака слов и сохранение в файл"""
        if not lemmas:
            print(f"Нет данных для создания облака слов для {title}")
            return
            
        text = ' '.join(lemmas)
        
        try:
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10,
                font_path=None
            ).generate(text)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=20, pad=20)
            plt.tight_layout(pad=0)
            
            safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            filename = f"wordcloud_{safe_title.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Облако слов сохранено: {filename}")
            
        except Exception as e:
            print(f"Ошибка при создании облака слов: {e}")
    
    def plot_frequency_comparison(self, freq_data, title):
        """Сравнительный график частот слов с сохранением в файл"""
        if not freq_data:
            print("Нет данных для сравнительного графика")
            return
            
        fig, axes = plt.subplots(1, len(freq_data), figsize=(18, 6))
        
        if len(freq_data) == 1:
            axes = [axes]
        
        for idx, (name, frequencies) in enumerate(freq_data.items()):
            if not frequencies:
                continue
            words, counts = zip(*frequencies[:20])
            axes[idx].barh(range(len(words)), counts, color='steelblue')
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Частота', fontsize=10)
            axes[idx].set_title(name, fontsize=12, wrap=True)
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        filename = "frequency_comparison.png"
        plt.savefig(filename)
        plt.close()
        print(f"Сравнительный график сохранен: {filename}")
    
    def analyze_book(self, url, title):
        """Анализ одного произведения"""
        print(f"\n{'='*60}")
        print(f"Анализ: {title}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        print("Загрузка текста...")
        text = self.fetch_text(url)
        if not text:
            print("Ошибка: не удалось загрузить текст")
            return None
        
        print(f"Загружено символов: {len(text):,}")
        
        if len(text) < 1000:
            print("Предупреждение: текст слишком короткий, возможно загрузилась не вся страница")
        
        print("Предобработка текста (токенизация, лемматизация)...")
        lemmas = self.preprocess_text(text)
        
        if len(lemmas) == 0:
            print("Ошибка: после обработки не осталось слов")
            return None
            
        print(f"Обработано слов (после лемматизации): {len(lemmas):,}")
        print(f"Уникальных слов: {len(set(lemmas)):,}")
        
        frequencies = self.get_word_frequency(lemmas, top_n=50)
        
        return {
            'title': title,
            'lemmas': lemmas,
            'frequencies': frequencies,
            'total_words': len(lemmas),
            'unique_words': len(set(lemmas))
        }
    
    def comparative_analysis(self, books_data):
        """Сопоставительный анализ произведений"""
        print(f"\n{'='*60}")
        print("СОПОСТАВИТЕЛЬНЫЙ АНАЛИЗ")
        print(f"{'='*60}\n")
        print("Общая статистика:")
        stats_df = pd.DataFrame([
            {
                'Произведение': book['title'],
                'Всего слов': f"{book['total_words']:,}",
                'Уникальных слов': f"{book['unique_words']:,}",
                'Лексическое разнообразие': f"{book['unique_words']/book['total_words']:.3f}"
            }
            for book in books_data
        ])
        print(stats_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("ТОП-10 НАИБОЛЕЕ ЧАСТОТНЫХ СЛОВ")
        print("="*60)
        for book in books_data:
            print(f"\n{book['title']}:")
            for i, (word, count) in enumerate(book['frequencies'][:10], 1):
                print(f"  {i:2d}. {word:15s} - {count:4d} раз")
    
        print("\n" + "="*60)
        print("ОБЩИЕ ЧАСТОТНЫЕ СЛОВА (встречаются во всех произведениях)")
        print("="*60)

        top_words_sets = [set(word for word, _ in book['frequencies'][:30]) for book in books_data]
        common_words = set.intersection(*top_words_sets) if top_words_sets else set()
        
        if common_words:
            print(f"Найдено общих слов: {len(common_words)}")
            print(", ".join(sorted(common_words)))
        else:
            print("Общих частотных слов не найдено в топ-30")    
        
        print("\n" + "="*60)
        print("ХАРАКТЕРНЫЕ СЛОВА (есть в топ-20 только одного произведения)")
        print("="*60)
        
        for book in books_data:
            book_top_words = set(word for word, _ in book['frequencies'][:20])
            other_books_words = set()
            for other_book in books_data:
                if other_book['title'] != book['title']:
                    other_books_words.update(word for word, _ in other_book['frequencies'][:50])
            
            unique_words = book_top_words - other_books_words
            if unique_words:
                print(f"\n{book['title']}:")
                print(", ".join(sorted(unique_words)))

def main():
    """Основная функция программы"""
    print("="*60)
    print("АНАЛИЗ ЧАСТОТЫ СЛОВ В ЛИТЕРАТУРНЫХ ПРОИЗВЕДЕНИЯХ")
    print("="*60)
    
    analyzer = LiteraryAnalyzer()

    books = [
        {
            'url': 'https://azbyka.ru/fiction/evgenij-onegin/',
            'title': 'Евгений Онегин (Пушкин)',
            'short_name': 'onegin'
        },
        {
            'url': 'https://azbyka.ru/fiction/idiot/',
            'title': 'Идиот (Достоевский)',
            'short_name': 'idiot'
        },
        {
            'url': 'https://azbyka.ru/fiction/mumu/',
            'title': 'Муму (Тургенев)',
            'short_name': 'mumu'
        }
    ]
    
    books_data = []
    for book in books:
        result = analyzer.analyze_book(book['url'], book['title'])
        if result:
            books_data.append(result)
            
            analyzer.create_wordcloud(
                result['lemmas'],
                result['title']
            )
    
    if not books_data:
        print("\nНе удалось проанализировать ни одно произведение")
        return

    if len(books_data) > 1:
        analyzer.comparative_analysis(books_data)
        
        freq_data = {book['title']: book['frequencies'] for book in books_data}
        analyzer.plot_frequency_comparison(
            freq_data,
            'Сравнение частоты слов в произведениях'
        )
    
    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)
    
    print("\nАнализ завершен! Все графики сохранены в директорию скрипта.")
    print("\nИспользованные произведения:")
    for book in books_data:
        print(f"  • {book['title']}: {book['total_words']:,} слов")

if __name__ == "__main__":
    main()