from locust import HttpUser, task, between
import string 
import random
import json
import os

# Initialization
HOST_URL = "13.234.232.198"
API_VERSION = "v1.0"
REGISTER_PORT = 8000
PREDICTION_PORT = 8001

class LoadTestModule(HttpUser):
    def __init__(self, parent):
        super(LoadTestModule, self).__init__(parent)
        self.token = None
    
    # wait_time = between(1, 2)

    def on_start(self):
        user_name = ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k = 7))    
        pass_word = ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k = 9))    
        credentials = {'username': user_name, 'password': pass_word}
        register = self.client.post(f'http://{HOST_URL}:{REGISTER_PORT}/{API_VERSION}/register',
                                 headers={'Content-type': 'application/json'},
                                 json=credentials)
        login_token = self.client.post(f'http://{HOST_URL}:{REGISTER_PORT}/{API_VERSION}/login',
                              headers={'Content-type': 'application/json'},
                              json=credentials)
        self.token = login_token.text[1:-1] 

    @task
    def prediction(self):
        path_to_txt = "../../data/transcripts/"
        path_to_json = "../../data/output/summary/"
        txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
        for index, tf in enumerate(txt_files):
            with open(os.path.join(path_to_txt, tf)) as txt_file:
                file_text = txt_file.read()
            with open(os.path.join(path_to_json, 'summary_' + tf[:-4] + '.json')) as json_file:
                json_text = json.load(json_file)
                summary_text = json_text['summary']
                result = self.client.post(f'http://{HOST_URL}:{PREDICTION_PORT}/{API_VERSION}/prediction',
                                headers={'Authorization': f'Bearer {self.token}', 'Content-type': 'application/json'},
                                json={'title': tf, 'model': 'all-mpnet-base-v2',
                                        'transcript': file_text, 'summary' : summary_text}
                                )
            print(result.json())