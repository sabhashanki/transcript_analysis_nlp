
# Importing function dependent packages
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, constr
import nltk
nltk.download('punkt')
import re

# Importing encryption libraries
from passlib.context import CryptContext

# Importing basic libraries
import jwt
import os
import configparser

# Importing environment modules
from dotenv import load_dotenv
load_dotenv()

# Importing fastapi libraries
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Configuration parser
config = configparser.ConfigParser()
config.read("./config_summarize.ini")
chunk_size = int(config['SUMMARIZE']['CHUNK_SIZE'])

# Authentication schema
class AuthDetails(BaseModel):
    username: constr(min_length=6, max_length=15)
    password: constr(min_length=8, max_length=15)

# Prediction module input scheme
class Item(BaseModel):
    title: str
    transcript: constr(min_length = 30)
    model: str

# Token generation, verification, Password hashing
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


# Tokenising the transcript
def preprocess_transcript(transcript_text, num_word_th=0):
    try:
        output_list = []
        assert(type(transcript_text)==list)
        for line in transcript_text:
            line_list = sent_tokenize(line)
            for l in line_list:
                if len(l) > num_word_th:
                    output_list.append(l)
        return output_list
    except Exception:
        raise HTTPException(status_code = 500, detail = 'PREPROCESSING ERROR')


# Creating transcript chunks 
def divide_chunks(l, n=chunk_size):
    try:
        for i in range(0, len(l), n):
            yield l[i:i + n]
    except Exception:
        raise HTTPException(status_code = 500, detail = 'DIVIDING CHUNKS ERROR')
