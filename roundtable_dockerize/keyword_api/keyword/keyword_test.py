
# Importing fastapi modules
from fastapi import FastAPI, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
# Importing keyword modules
from utils_key import Item, AuthHandler, final_processing, over_all_key, divide_chunks, NER_transcript, nounKey_nerKey_summary_chunk, clean_transcript
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk.tokenize import sent_tokenize
import spacy
# Importing basic modules
import datetime
import logging
import sys
# Importing configuration modules
import configparser


logging.basicConfig(level = logging.INFO, filename = 'keyword.log',
                    filemode = 'w', format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI()


# Module initialization
try:
    auth_handler = AuthHandler()
    config = configparser.ConfigParser()
    config.read("./config_key.ini")
    model_name = config['KEYWORD']['MODEL_NAME']
    nlp = spacy.load('en_core_web_md')
except Exception:
    logging.exception(sys.exc_info())
    raise HTTPException(status_code = 500, detail = 'PRELIMINARY INITIALIZATION FAILED')

try:
    kw_extractor = KeyBERT('/home/ms/project/models/roundtable/all-mpnet-base-v2')  # Place the files in the same folder
except Exception:
    logging.exception(sys.exc_info())
    raise HTTPException(status_code = 500, detail = 'MODEL INITIALIZATION FAILED')


# Prediction Endpoint
@app.post("/v1.0/prediction")
async def read_text(data: Item, username = Depends(auth_handler.auth_wrapper)):
    from timeit import default_timer as timer
    st = timer()
    print(f'start: {st}')
    if data.model == model_name:
        time = datetime.datetime.now()
        
        logging.info('Executing keyword module')
        summary_text = data.summary
        transcript_text = data.transcript

        # clean transcript
        transcript_text = clean_transcript(transcript_text)
        
        try:
            summary_line_list = sent_tokenize(summary_text)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code=500, detail='SENTENCE TOKENIZING ERROR')

        # Dividing into chunks    
        summary_line_chunks_list = divide_chunks(summary_line_list)
        summary_chunks_txt_list = [' '.join(i) for i in summary_line_chunks_list]
        divide = timer()
        result = divide-st
        print(f'after divide junks: {result}')
        try:
            keybert_diversity_phrases = []
            NER_Keywords = []
            noun_keywords = []
            for new_text in summary_chunks_txt_list:
                try:
                    keywords_n = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.*>'), use_mmr=True, diversity=1.0,
                                                                keyphrase_ngram_range=(1, 1), stop_words='english', top_n=50)
                    keywords_noun = [i for i in keywords_n if i[1] > 0.2]
                    for i, _ in keywords_noun:
                        keybert_diversity_phrases.append(i)
                except:
                    logging.info('No keywords extracted in this loop')

                try:
                    keywords2_nn = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.+>+<N.+>'), use_mmr=True, diversity=1.0,
                                                                    keyphrase_ngram_range=(2, 3), stop_words='english', top_n=50)
                    keywords_nnounn = [i for i in keywords2_nn if i[1] > 0.2]
                    for i, _ in keywords_nnounn:
                        keybert_diversity_phrases.append(i)
                except:
                    logging.info('No keywords extracted in this loop')

                # Extrating noun and NER from chunk
                noun_keywords, NER_Keywords = nounKey_nerKey_summary_chunk(new_text, noun_keywords, NER_Keywords)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code=500, detail='KEYWORD EXTRACTION ERROR')

        # Overall keywords
        over_all_keywords, noun_keywords, ner_keywords = over_all_key(keybert_diversity_phrases, NER_Keywords, noun_keywords)

        keybert_out = timer()
        print(f'after keybert: {keybert_out - divide}')

        # keywords from transcript
        over_all_keywords, ner_keywords_trans = NER_transcript(transcript_text, over_all_keywords, ner_keywords)
        # postprocessing final keywords
        over_all_keywords, noun_keywords_all, ner_keywords_all = final_processing(over_all_keywords, ner_keywords_trans, noun_keywords, keybert_diversity_phrases)

        trans_NER_final = timer()
        print(f'after NERtrans_preprocessing: {trans_NER_final - keybert_out}')

        try:
            # returning the result
            content = {'Title' : data.title, 'NER Keywords' : ner_keywords_all, 
                       'Noun Keywords' : noun_keywords_all, 'Over_all_keywords' : over_all_keywords, 
                       'timestamp' : time
                       }
            content = jsonable_encoder(content)
            end_point = timer() 
            print(f'ending: {end_point - st}')
            return JSONResponse(content)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code=500, detail='ERROR IN RETURNING THE OUTPUT')
    else:
        raise HTTPException(status_code=404, detail='INVALID MODEL NAME - AVAILABLE MODEL : <all-mpnet-base-v2>')
