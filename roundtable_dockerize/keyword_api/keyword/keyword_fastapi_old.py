# Importing fastapi modules
from fastapi import FastAPI, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
# Importing keyword modules
from utils_key import Item, AuthHandler, divide_chunks, NER_transcript, clean_transcript
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import spacy
# Importing basic modules
import datetime
import logging
import sys
import re
import string
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
    kw_extractor = KeyBERT('sentence-transformers/all-mpnet-base-v2')
    nlp = spacy.load('en_core_web_md')
except Exception:
    raise HTTPException(status_code = 500, detail = 'INITIALIZATION FAILED')


# Prediction Endpoint
@app.post("/v1.0/prediction")
async def read_text(data: Item, username = Depends(auth_handler.auth_wrapper)):
    if data.model == model_name:
        time = datetime.datetime.now()
        try:
            logging.info('Executing keyword module')
            summary_text = data.summary
            transcript_text = data.transcript

            # Cleaning the transcript
            summary_text = clean_transcript(summary_text)

            # get_keywords_from_summary -> keyword_extractor.py
            summary_line_list = sent_tokenize(summary_text)
            summary_line_chunks_list = divide_chunks(summary_line_list)
            summary_chunks_txt_list = [' '.join(i) for i in summary_line_chunks_list]

            # predict -> keyword_extractor.py
            keybert_diversity_phrases = []
            NER_Keywords = []
            noun_keywords = []
            for new_text in summary_chunks_txt_list:
                try:
                    keywords_n = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.*>'), use_mmr=True, diversity=1.0,
                                                                keyphrase_ngram_range=(1, 1), stop_words='english', top_n=50)
                    # print(keywords_n)
                    keywords_noun = [i for i in keywords_n if i[1] > 0.2]
                    for i, _ in keywords_noun:
                        keybert_diversity_phrases.append(i)
                except:
                    print('no keywords found for this loop-a')
                try:
                    keywords2_nn = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.+>+<N.+>'), use_mmr=True, diversity=1.0,
                                                                    keyphrase_ngram_range=(2, 3), stop_words='english', top_n=50)
                    keywords_nnounn = [i for i in keywords2_nn if i[1] > 0.2]
                    for i, _ in keywords_nnounn:
                        keybert_diversity_phrases.append(i)
                except:
                    print('no keywords found for this loop-b')
                
                #extract ner keywords
                doc = nlp(new_text)
                for noun_chunk in doc.noun_chunks:
                    text = re.sub(r"[%,:/!@#$^&*()+=|_'?><,.`~-]", "", str(noun_chunk))  
                    text = text.strip() 
                    if text.isnumeric():
                        print(f'{noun_chunk} is a number/punctuation')
                        continue
                    if noun_chunk.root.pos_ in ["PROPN", "NOUN"]:
                        if not any(word.is_stop for word in noun_chunk):
                            noun_keywords.append(noun_chunk.text)
                        elif (not any(word.is_stop for word in noun_chunk[1:])) and noun_chunk.text.split(" ")[0].lower() == "the":
                            noun_keywords.append(noun_chunk.text) 
                        else:
                            print(f"filtered any stop words: {noun_chunk.text}")
                        
                    else:
                        print(f"filtering {noun_chunk.text}")
                
                for ent in doc.ents:
                    if ent.label_ in [ 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']:
                        NER_Keywords.append(f"{ent.text}_({ent.label_.lower()})")
            NER_Keywords = [word for word in NER_Keywords if word not in string.punctuation]
            keywords = list(set([i.lower() for i in keybert_diversity_phrases]))
            ner_keywords = list(set([i.lower() for i in NER_Keywords]))
            ner_keywords = [nr.split("_")[0] for nr in ner_keywords]
            noun_keywords = list(set([i.lower() for i in noun_keywords]))
            print(noun_keywords)

            
            # run.py
            over_all_keywords = list(set(keywords + ner_keywords + noun_keywords))
        
            # score_keyword -> keyword_extractor.py
            if len(over_all_keywords) > 0:
                sen_encode = kw_extractor.model.embed(summary_text)
                keyword_encode = kw_extractor.model.embed(over_all_keywords)
                score = cosine_similarity(sen_encode.reshape(1, -1), keyword_encode)
                score = [int(s*1000) for s in score[0]]
                over_all_keywords = sorted(zip(over_all_keywords, score), key = lambda x : x[1], reverse=True)
            over_all_keywords = [kw[0] for kw in over_all_keywords]
            over_all_keywords, ner_keywords_trans = NER_transcript(transcript_text, over_all_keywords, ner_keywords)
            
            # result_visualizer.py
            ner_keywords_trans = [nr.split("_")[0] for nr in ner_keywords_trans]
            noun_keywords = set(set(over_all_keywords) - set(ner_keywords_trans)) - set(keywords)
            ner_keywords_trans = ", ".join(ner_keywords_trans)
            over_all_keywords = ", ".join(over_all_keywords)

            content = {'Title' : data.title, 'NER Keywords' : ner_keywords_trans, 
                       'Noun Keywords' : noun_keywords, 'Over_all_keywords' : over_all_keywords, 
                       'timestamp' : time
                       }
            
            content = jsonable_encoder(content)
            return JSONResponse(content)
        except Exception:
            logging.exception(sys.exc_info())
            raise HTTPException(status_code=500, detail='MODULE EXECUTION ERROR')
    else:
        raise HTTPException(status_code=404, detail='INVALID MODEL NAME - AVAILABLE MODEL : <all-mpnet-base-v2>')