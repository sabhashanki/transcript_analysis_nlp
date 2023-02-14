
# Importing fastapi libraries
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# Importing basic libraries
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, constr
import os
import jwt
# Importing encryption libraries
from passlib.context import CryptContext

# Initialization
load_dotenv()


# Authentication and password hashing
class AuthHandler():
    security = HTTPBearer()
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    secret = os.getenv('AUTH_KEY')

    def get_password_hash(self, password):
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)

    def encode_token(self, user_id):
        payload = {
            'exp': datetime.utcnow() + timedelta(days=0, minutes=20),
            'iat': datetime.utcnow(),
            'sub': user_id
        }
        return jwt.encode(
            payload,
            self.secret,
            algorithm='HS256'
        )

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


# schema for the authentication
class AuthDetails(BaseModel):
    username: constr(min_length=6, max_length=15)
    password: constr(min_length=8, max_length=15)
