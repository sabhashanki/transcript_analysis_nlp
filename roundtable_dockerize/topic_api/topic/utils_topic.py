
# Importing encryption libraries
from passlib.context import CryptContext
# Importing basic libraries
import jwt
import os
# Importing dependent modules
import numpy as np
import math
from pydantic import BaseModel, constr
from scipy.special import softmax
# Importing environment modules
from dotenv import load_dotenv
load_dotenv()
# Importing fastapi libraries
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Prediction endpoint input scheme
class Item(BaseModel):
    title: str
    summary: constr(min_length=20)
    model: str
    
    
# Password Hashing, token generation and validation
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


# calculate similarity score - v1 to v2: (v1 dot v2)/{||v1||*||v2||)
def cosine_similarity(v1, v2):
    try:
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)
    except Exception:
        raise HTTPException(status_code = 500, detail = 'COSINE SIMILARITY CALCULATION FAILED')

# Getting similarity score for text and topic
def get_similarities_model1(embed_text, embed_topic):
    try:
        SCALE_SCORE = 20
        scored = []
        for i in range(len(embed_topic)):
            for j in range(len(embed_text)):
                score = np.round(cosine_similarity((embed_text[j]), (embed_topic[i])), 3)
                scored.append(score * SCALE_SCORE)
        scored = softmax(np.array(scored))
        return scored
    except Exception:
        raise HTTPException(status_code = 500, detail = 'RESULT SCORING FAILED')


