from pyaspeller import YandexSpeller
from flask import Flask, request, jsonify
from flask_cors import CORS
import zipfile
import wget
import gensim
import pymorphy2

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from flask_caching import Cache

app = Flask(__name__)


CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"] 
)

talisman = Talisman(app)

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
    return (dict[part_of_speech])

def cos_distance(word1, word2):
    word1 = word1 + "_" + part_speech(word1)
    word2 = word2 + "_" + part_speech(word2)
    model_file = '180.zip'
    with zipfile.ZipFile(model_file, 'r') as archive:
        stream = archive.open('model.bin')
        model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    dist = model.similarity(word1, word2)
    return str(dist)



@app.route('/get-csrf-token', methods=['GET'])
def get_csrf_token():
    csrf_token = generate_csrf()
    print(csrf_token)
    return jsonify({'csrf_token': csrf_token})

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

@app.route('/compare', methods=['POST'])
@limiter.limit("10 per minute")  
@cache.cached(timeout=60)  
def compare_words():
    data = request.get_json()
    word1 = data.get('word1')
    word2 = data.get('word2')

    word1 = correct(word1)
    word2 = correct(word2)
    number = cos_distance(word1, word2)
    

    return jsonify({
        'word1': word1,
        'word2': word2,
        'number': number,
    })

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=True) 