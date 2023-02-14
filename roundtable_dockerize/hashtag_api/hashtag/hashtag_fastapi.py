
# Importing fastapi modules
from fastapi import FastAPI, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
# Importing keyword modules
from utils import Item, AuthHandler
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import spacy
from nltk.tokenize import sent_tokenize
from utils import divide_chunks, camel
# Importing basic modules
import datetime
import logging
import sys
# Importing configuration modules
import configparser


logging.basicConfig(level = logging.INFO, filename = 'hashtag.log',
                    filemode = 'w', format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI()


# Module initialization
try:
    logging.info('Modules initialization')
    # Authentication module
    auth_handler = AuthHandler()
    # Configuration Parser
    config = configparser.ConfigParser()
    config.read("./config_hashtag.ini")
    model_name = config['HASHTAG']['MODEL_NAME']
    nlp = spacy.load('en_core_web_md')
except Exception:
    logging.exception(sys.exc_info())
    raise HTTPException(status_code = 500, detail = 'PRELIMINATY INITIALIZATION FAILED')

try:
    # Keyword module 
    kw_extractor = KeyBERT('/home/ms/project/models/roundtable/all-mpnet-base-v2')  # Place the files in the same folder
except Exception:
    logging.exception(sys.exc_info())
    raise HTTPException(status_code = 500, detail = 'MODEL INITIALIZATION FAILED')


# Prediction Endpoint
@app.post("/v1.0/prediction")
async def read_text(data: Item, username = Depends(auth_handler.auth_wrapper)):

    logging.info('Executing prediction endpoint')
    if data.model == model_name:
        time = datetime.datetime.now()

        try:   
            logging.info('Tokenizing summary')
            summary_line_list = sent_tokenize(data.summary)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code = 500, detail = 'ERROR IN TOKENIZING SENTENCES')


        logging.info('Dividing into chunks')
        summary_line_chunks_list = divide_chunks(summary_line_list)
        summary_chunks_txt_list = [' '.join(i) for i in summary_line_chunks_list]


        try:
            logging.info('Executing keyword module')
            keybert_diversity_phrases = []
            for new_text in summary_chunks_txt_list:
                try:
                    logging.info('Executing Noun phrase loop')
                    keywords_n = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.*>'), use_mmr=True, diversity=1.0,
                                                                keyphrase_ngram_range=(1, 1), stop_words='english', top_n=50)
                    keywords_noun = [i for i in keywords_n if i[1] > 0.2]
                    for i, _ in keywords_noun:
                        keybert_diversity_phrases.append(i)
                except:
                    logging.info('No keywords extracted in this loop')
                try:
                    logging.info('Executing Noun-Noun phrase loop')
                    keywords2_nn = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.+>+<N.+>'), use_mmr=True, diversity=1.0,
                                                                    keyphrase_ngram_range=(2, 3), stop_words='english', top_n=50)
                    keywords_nnounn = [i for i in keywords2_nn if i[1] > 0.2]
                    for i, _ in keywords_nnounn:
                        keybert_diversity_phrases.append(i)
                except:
                    logging.info('No keywords extracted in this loop')
            keywords = list(set([i.lower() for i in keybert_diversity_phrases]))
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code = 500, detail = 'ERROR IN EXTRACTING KEYWORDS')


        logging.info('Executing camelcase module')
        camelcase_keywords = list(set(camel(keywords)))


        try:
            logging.info('Post Processing')
            hashtag_print = ", ".join([f"#{kw}" for kw in camelcase_keywords])
            content = {'Title' : data.title, 'Hashtag' : hashtag_print, 'timestamp' : time}
            content = jsonable_encoder(content)
            logging.info('returning output')
            return JSONResponse(content)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code = 500, detail = 'ERROR IN POST PROCESSING AND RETURNING THE OUTPUT')
    else:
        raise HTTPException(status_code = 404, detail = 'INVALID MODEL NAME - AVAILABLE MODEL : <all-mpnet-base-v2>')
