
# Importing authentication and encryption libraries
from passlib.context import CryptContext
from pydantic import BaseModel, constr
# Importing basic libraries
import jwt
import os
import configparser
import nltk
nltk.download('punkt')
# Importing environment modules
from dotenv import load_dotenv
load_dotenv()
# Importing fastapi libraries
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Configuration Parser
config = configparser.ConfigParser()
config.read("./config_hashtag.ini")
chunk_size = int(config['HASHTAG']['CHUNK_SIZE'])


# Prediction module input scheme
class Item(BaseModel):
    title: str
    summary: constr(min_length = 30)
    model: str


# Password hashing, Token generation and validation
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


# Dividing the summary into chunks
def divide_chunks(l, n=chunk_size):
    try:
        for i in range(0, len(l), n):
            yield l[i:i + n]
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN DIVIDE_CHUNKS MODULE')


# CamelCase formatting
def camel(keywords):
    try:
        final = []
        for text in keywords:
            words = text.split(' ')
            if len(words) > 2:
                words = "".join([word.title() for word in words])
                final.append(words)
            else:
                final.append(words[0])
        return final
    except Exception:
        raise HTTPException(status_code = 500, detail = 'ERROR IN CAMEL_CASE MODULE')