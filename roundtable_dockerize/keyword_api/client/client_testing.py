
# Importing basic modules
import requests
import time
import os
from statistics import mean
import json

# Initialization
HOST_URL = "localhost"
API_VERSION = "v1.0"
PREDICTION_PORT = 8001
REGISTER_PORT = 8000
credentials = {'username': 'khulke', 'password': 'aifylabs'}


def client_script(transcript_text, summary_text, title_text):
    # Register Module
    try:
        register = requests.post(f'http://{HOST_URL}:{REGISTER_PORT}/{API_VERSION}/register',
                                 headers={'Content-type': 'application/json'},
                                 json=credentials
                                 )
        if register.status_code == 400:
            print(f'USER ALREADY REGISTERED - TRY DIFFERENT USERNAME')
        elif register.status_code == 422:
            print(f'USERNAME(FORMAT:STRING, MIN:6, MAX:15) OR PASSWORD(FORMAT:STRING, MIN:8, MAX:15) DOES NOT MEET THE REQUIREMENTS')
        elif register.status_code == 201:
            print(f'USERNAME: {credentials["username"]} CREATION : SUCCESS')
    except:
        return 'REGISTRATION ERROR - TRY AGAIN'

    # Login Module
    try:
        token = requests.post(f'http://{HOST_URL}:{REGISTER_PORT}/{API_VERSION}/login',
                              headers={'Content-type': 'application/json'},
                              json=credentials
                              )
        if token.status_code == 401:
            print('INVALID USERNAME/PASSWORD')
        elif token.status_code == 500:
            print('TOKEN GENERATION FAILED')
        elif token.status_code == 201:
            token = token.text[1:-1]
            print(f'USER : {credentials["username"]} LOGGED IN - TOKEN VALID FOR 20 MINUTES')
    except:
        return 'LOGIN ERROR'

    # Prediction Module
    try:
        st = time.time()
        result = requests.post(f'http://{HOST_URL}:{PREDICTION_PORT}/{API_VERSION}/prediction',
                                headers={'Authorization': f'Bearer {token}', 'Content-type': 'application/json'},
                                json={'title': title_text, 'model': 'all-mpnet-base-v2',
                                        'transcript': transcript_text, 'summary' : summary_text}
                                )
        et = time.time()
        if result.status_code == 201 or result.status_code == 200:
            print(result.json())
            print(f'Inference Time : {et-st}')
        elif result.status_code == 422:
            print('INVALID INPUT - TITLE, SUMMARY(MIN 20 CHARACTERS) AND MODEL NAME SHOULD BE IN STRING FORMAT')
        elif result.status_code == 401:
            print('TOKEN EXPIRED')
        elif result.status_code == 498:
            print('INVALID TOKEN')
        elif result.status_code == 500:
            print('MODULE EXECUTION ERROR')
        elif result.status_code == 404:
            print('INVALID MODEL NAME')
    except:
        print('CLIENT MODULE EXECUTION ERROR')

    return et-st



time_list = []

path_to_txt = "../../../data/transcripts/"
path_to_json = "../../../data/output/summary/"
txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
for index, tf in enumerate(txt_files):
    with open(os.path.join(path_to_txt, tf)) as txt_file:
        file_text = txt_file.read()
    with open(os.path.join(path_to_json, 'summary_' + tf[:-4] + '.json')) as json_file:
        json_text = json.load(json_file)
        summary_text = json_text['summary']
    time_list.append(client_script(file_text, summary_text, tf))

print(f'Overall Inference Time: {mean(time_list)}')