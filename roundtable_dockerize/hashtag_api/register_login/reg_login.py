
# Importing fastapi modules
from fastapi import FastAPI, HTTPException
# Importing dependent modules for authentication and schemas
from utils import AuthHandler, AuthDetails
# Importing basic modules
import logging
import pandas as pd
import sys


logging.basicConfig(level=logging.INFO, filename='reg_login.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI()

# Module Initialization
try:
    auth_handler = AuthHandler()
    logging.info('Initialization Done')
except Exception:
    logging.exception(sys.exc_info())
    raise HTTPException(status_code=500, detail='INITIALIZATION FAILED')


# Registration Endpoint
@app.post('/v1.0/register', status_code=201)
async def register(auth_details: AuthDetails):
    df = pd.read_json('db.json', lines=True)
    if any(x == auth_details.username for x in df.username):
        raise HTTPException(status_code=400, detail=f'USER ALREADY REGISTERED - TRY DIFFERENT USERNAME')
    logging.info('Username availability check')
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    logging.info('Password Hashing')
    df.loc[len(df.index)] = [auth_details.username, hashed_password]
    df.to_json('db.json', orient='records', lines=True)
    return f'REGISTRATION SUCCESSFUL: {bool(1)}'


# Login Endpoint
@app.post('/v1.0/login', status_code=201)
async def login(auth_details: AuthDetails):
    df = pd.read_json('db.json', lines=True)
    user = None
    logging.info('Username Authentication')
    for index, row in df.iterrows():
        if row['username'] == auth_details.username:
            user = row['username']
            password = row['password']
            break
    if (user is None) or (not auth_handler.verify_password(auth_details.password, password)):
        raise HTTPException(status_code=401, detail='INVALID USERNAME/PASSWORD')
    try:
        token = auth_handler.encode_token(user)
        logging.info('token generation')
    except Exception:
        logging.exception(sys.exc_info())
        raise HTTPException(status_code=500, detail='TOKEN GENERATION FAILED')
    return token
