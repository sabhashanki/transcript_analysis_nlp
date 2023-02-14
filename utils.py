import re
from nltk.tokenize import sent_tokenize, word_tokenize

def divide_chunks(l, n=30):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def remove_timestamp(text_lines):
    assert(type(text_lines)==list)
    regex = re.compile('[^a-zA-Z]')
    text_list = []
    for line in text_lines:
        if len(line) < 30:
            tmp_line = regex.sub('', line)
            if len(tmp_line) > 2:
                text_list.append(line)
        else:
              text_list.append(line)
    return text_list

def preprocess_transcript(transcript_text, num_word_th=0):
    output_list = []
    assert(type(transcript_text)==list)
    for line in transcript_text:
        line_list = sent_tokenize(line)
        for l in line_list:
            if len(l) > num_word_th:
                output_list.append(l)
    return output_list

def read_transcript(fpath):
    with open(fpath,"r") as fp:
        transcript_txt = fp.read()
    transcript_txt_lines = transcript_txt.split("\n")
    #cleanup lines which are timestamps or number of words are less than 5-6.
    transcript_txt_lines = remove_timestamp(transcript_txt_lines)
    return transcript_txt_lines