from locust import HttpUser, task, between
import string 
import random
import json
import os

# Initialization
HOST_URL = "3.109.201.89"
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
        txt_files = [pos_txt for pos_txt in os.listdir(path_to_txt) if pos_txt.endswith('.txt')]
        for index, tf in enumerate(txt_files):
            with open(os.path.join(path_to_txt, tf)) as txt_file:
                file_text = txt_file.read()
                result = self.client.post(f'http://{HOST_URL}/{API_VERSION}/prediction',
                                headers={'Authorization': f'Bearer {self.token}', 'Content-type': 'application/json'},
                                json={'title': tf, 'model': 'bart-large-cnn',
                                        'transcript': file_text}
                                )
            print(result.json())