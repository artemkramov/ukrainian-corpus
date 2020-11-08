from flask import Flask, request, jsonify, abort
from coreference_ua import CoreferenceUA
from coherence_ua.transformer_coherence import CoherenceModel
import noun_phrase_ua
app = Flask(__name__)

print('Init coherence model...')
model_coherence = CoherenceModel()
print('Init coreference model...')
model_coreference = CoreferenceUA()
print('Init phrase extractor...')
model_phrase = noun_phrase_ua.NLP()

print('Ready to accept queries')

def get_text_from_request():
    content = request.json
    if (not (content is None)) and 'text' in content:
        return content['text']
    abort(400)
    
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/get_coreferent_clusters', methods=['POST'])
def get_coreferent_clusters():
    text = get_text_from_request()
    return jsonify(model_coreference.set_text(text).extract_phrases())
    
@app.route('/api/get_phrases', methods=['POST'])
def get_phrases():
    text = get_text_from_request()
    summary = model_phrase.extract_entities(text)
    summary['entities'] = list(summary['entities'])
    return jsonify(summary)
    
@app.route('/api/get_coherence', methods=['POST'])
def get_coherence():
    text = get_text_from_request()
    summary = {
        "series": model_coherence.get_prediction_series(text),
        "coherence_product": model_coherence.evaluate_coherence_as_product(text),
        "coherence_threshold": model_coherence.evaluate_coherence_using_threshold(text, 0.1)
    }
    return jsonify(summary)
    