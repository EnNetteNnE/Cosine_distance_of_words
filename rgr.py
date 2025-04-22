from pyaspeller import YandexSpeller
from flask import Flask, request, jsonify
from flask_cors import CORS
import zipfile
import wget
import gensim
import pymorphy2
import requests

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from pymorphy2 import MorphAnalyzer

import psycopg2
from psycopg2 import sql


app = Flask(__name__)


CORS(app, resources={r"/*": {"origins": "*"}})


DB_CONFIG = {
    'dbname': 'rgr',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

def get_antonym(word):
    conn = get_db_connection()
    if not conn:
        return "не найдено"
    try:
        with conn.cursor() as cursor:
            query = sql.SQL("""
                SELECT CASE 
                    WHEN word1 = %s THEN word2
                    WHEN word2 = %s THEN word1
                END as antonym
                FROM antonyms
                WHERE word1 = %s OR word2 = %s
                LIMIT 1
            """)
            cursor.execute(query, (word, word, word, word))
            result = cursor.fetchone()         
            return result[0] if result else "не найдено"
    except psycopg2.Error as e:
        print(f"Ошибка при поиске антонима: {e}")
        return "не найдено"
    finally:
        conn.close()

def find_sentence_with_word(word):
    conn = get_db_connection()
    if not conn:
        return "не найдено"
    try:
        with conn.cursor() as cursor:
            query = sql.SQL("""
                SELECT suggestion
                FROM context
                WHERE suggestion ILIKE %s
                LIMIT 1
            """)
            cursor.execute(query, (f'%{word}%',))
            result = cursor.fetchone()           
            return result[0] if result else "не найдено"
    except psycopg2.Error as e:
        print(f"Ошибка при поиске предложения: {e}")
        return "не найдено"
    finally:
        conn.close()



limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"] 
)

def make_cache_key(*args, **kwargs):
    data = request.get_json()
    return f"{request.path}_{data.get('word1')}_{data.get('word2')}"

def correct(text):
    speller = YandexSpeller()
    changes = {change['word']: change['s'][0] for change in speller.spell(text)}
    for word, suggestion in changes.items():
        text = text.replace(word, suggestion)
    return text


def part_speech(word):
    dict = {
        "NOUN": "NOUN",
        "ADJF": "ADJ",
        "ADJS": "ADJ",
        "COMP": "ADJ",
        "VERB": "VERB",
        "INFN": "VERB",
        "PRTF": "NOUN",
        "PRTS": "NOUN",
        "GRND": "NOUN",
        "NUMR": "NUM",
        "ADVB": "ADV",
        "NPRO": "NOUN",
        "PRED": "NOUN",
        "PREP": "NOUN",
        "CONJ": "NOUN",
        "PRCL": "NOUN",
        "INTJ": "INTJ"
    }
    m2 = pymorphy2.MorphAnalyzer()
    parses = m2.parse(word)
    first_parse = parses[0]
    part_of_speech = first_parse.tag.POS
    return dict.get(part_of_speech, "NOUN")

def normalform(word):
    word = word.split('_')[0]
    m2 = pymorphy2.MorphAnalyzer()
    parses = m2.parse(word)
    lemma = parses[0].normal_form
    pos = part_speech(word)
    return f"{lemma}_{pos}"


def cos_distance(word1, word2):
    word1 = word1 + "_" + part_speech(word1)
    word2 = word2 + "_" + part_speech(word2)

    model_file = '180.zip'
    with zipfile.ZipFile(model_file, 'r') as archive:
        archive.extract('model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)

    if word1 not in model.key_to_index:
        word1 = normalform(word1)
    if word2 not in model.key_to_index:
        word2 = normalform(word2)

    dist = model.similarity(word1, word2)
    return str(dist)


def get_synonyms(word):
    
    api_key = "" 
    url = f"https://dictionary.yandex.net/api/v1/dicservice.json/lookup?key={api_key}&lang=ru-ru&text={word}"
    response = requests.get(url).json()
    
    synonyms = []
    
    if 'def' in response:
        for definition in response['def']:
            for tr in definition.get('tr', []):
                synonyms.extend([syn['text'] for syn in tr.get('syn', [])])
    
    if synonyms:
        result = ', '.join(synonyms)
    else:
        result = 'не найдено'
    
    return result


morph = MorphAnalyzer()
def get_related_words(word):
    try:
        parsed = morph.parse(word)[0]
        related = set()
        
        for lexeme in parsed.lexeme:
            related.add(lexeme.word)
        
        related.discard(word)
        
        return ', '.join(related) if related else "не найдено"
    
    except Exception as e:
        return "не найдено"
    

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

@app.route('/compare', methods=['POST'])
@limiter.limit("10 per minute")  
@cache.cached(timeout=60, key_prefix=make_cache_key)
def compare_words():
    data = request.get_json()
    if not data or 'word1' not in data or 'word2' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    word1 = data.get('word1')
    word2 = data.get('word2')

    word1 = correct(word1)
    word2 = correct(word2)

    synonyms1 = get_synonyms(word1)
    synonyms2 = get_synonyms(word2)

    antonym1 = get_antonym(word1)
    antonym2 = get_antonym(word2)

    context1 = find_sentence_with_word(word1)
    context2 = find_sentence_with_word(word2)

    related1 = get_related_words(word1)
    related2 = get_related_words(word2)

    try:
        number = cos_distance(word1, word2)
    except Exception as e:
        number = "не найдено"

    return jsonify({
        'word1': word1,
        'word2': word2,
        'number': number,
        'synonyms1': synonyms1,
        'synonyms2': synonyms2,
        'related1': related1,
        'related2': related2,
        'antonym1': antonym1,
        'antonym2': antonym2,
        'context1': context1,
        'context2': context2,
    })



if __name__ == '__main__':
    app.run(debug=True) 


