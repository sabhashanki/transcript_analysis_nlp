
# Importing function dependent libraries
from pydantic import BaseModel, constr
import spacy
# Importing encryption libraries
from passlib.context import CryptContext
# Importing basic libraries
import jwt
import os
import re
import configparser
import string
# Importing environment modules
from dotenv import load_dotenv
# Importing fastapi libraries
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Initialization
load_dotenv()
nlp = spacy.load('en_core_web_md')


# Configuration Parser
config = configparser.ConfigParser()
config.read("./config_key.ini")
chunk_size = int(config['KEYWORD']['CHUNK_SIZE'])


# Prediction module input scheme
class Item(BaseModel):
    title: str
    summary: constr(min_length = 30)
    transcript: constr(min_length = 30)
    model: str


class AuthHandler():
    security = HTTPBearer()
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    secret = os.getenv('AUTH_KEY')

    def decode_token(self, token):
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail='SIGNATURE HAS EXPIRED')
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=498, detail='INVALID TOKEN')

    def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Security(security)):
        return self.decode_token(auth.credentials)


# transcript cleaning - removing timestamp and newline character
def clean_transcript(transcript_text):
    try:
        regex = r"\d+.\d+-\d+.\d+"
        removed_timestamp = re.sub(regex, "", transcript_text)
        cleaned_text = removed_timestamp.replace("\n", "")
        cleaned_text = cleaned_text.replace("\\", "")
        return cleaned_text
    except Exception:
        raise HTTPException(status_code = 500, detail = 'TRANSCRIPT CLEANING ERROR')
    

# Dividing the summary into chunks
def divide_chunks(l, n=chunk_size):
    try:
        for i in range(0, len(l), n):
            yield l[i:i + n]
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN DIVIDE_CHUNKS MODULE')


# Extracting NER and Noun keywords from summary chunk
def nounKey_nerKey_summary_chunk(summary_chunk, noun_keywords, NER_Keywords):
    try:
        doc = nlp(summary_chunk)
        for noun_chunk in doc.noun_chunks:
            text = re.sub(r"[%,:/!@#$^&*()+=|_'?><,.`~-]", "", str(noun_chunk))  
            text = text.strip() 
            if text.isnumeric():
                continue
            if noun_chunk.root.pos_ in ["PROPN", "NOUN"]:
                if not any(word.is_stop for word in noun_chunk):
                    noun_keywords.append(noun_chunk.text)
                elif (not any(word.is_stop for word in noun_chunk[1:])) and noun_chunk.text.split(" ")[0].lower() == "the":
                    noun_keywords.append(noun_chunk.text) 

        for ent in doc.ents:
            if ent.label_ in [ 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']:
                NER_Keywords.append(f"{ent.text}_({ent.label_.lower()})")

        NER_Keywords = [word for word in NER_Keywords if word not in string.punctuation]
        return noun_keywords, NER_Keywords
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN CHUNK_KEYWORDS MODULE')


def over_all_key(keybert_diversity_phrases, NER_Keywords, noun_keywords):
    try:
        keywords = list(set([i.lower() for i in keybert_diversity_phrases]))
        ner_keywords = list(set([i.lower() for i in NER_Keywords]))
        ner_keywords = [nr.split("_")[0] for nr in ner_keywords]
        noun_keywords = list(set([i.lower() for i in noun_keywords]))
        over_all_keywords = list(set(keywords + ner_keywords + noun_keywords))
        return over_all_keywords, noun_keywords, ner_keywords
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN OVER_ALL_KEYWORDS MODULE')


# Keywords from transcript
def NER_transcript(transcript_text, over_all_keywords, ner_keywords):        
    try:
        doc = nlp(transcript_text)
        for ent in doc.ents:
            if ent.label_ in [ 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']:
                if ent.text not in ner_keywords:
                    ner_keywords.append(f"{ent.text}")

        for nr in ner_keywords:
            if nr not in over_all_keywords:
                over_all_keywords.insert(0, nr)
        return over_all_keywords, ner_keywords
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN TRANSCRIPT_KEYWORDS MODULE')


def final_processing(over_all_keywords, ner_keywords_trans, noun_keywords, keywords):
    try:
        ner_keywords_trans = [nr.split("_")[0] for nr in ner_keywords_trans]
        noun_keywords = set(set(over_all_keywords) - set(ner_keywords_trans)) - set(keywords)
        ner_keywords_trans = ", ".join(ner_keywords_trans)
        over_all_keywords = ", ".join(over_all_keywords)
        return over_all_keywords, noun_keywords, ner_keywords_trans
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN FINAL_PROCESSING MODULE')




