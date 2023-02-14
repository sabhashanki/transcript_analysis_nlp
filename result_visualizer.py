import streamlit as st
from annotated_text import annotated_text

from pathlib import Path
from utils import read_transcript
from run import get_summary, get_keyword

from keyword_extractor import nlp

from nltk.tokenize import sent_tokenize

from fuzzysearch import find_near_matches
from fuzzywuzzy import process

from topic_detector import get_topics, get_topic_gpt3, get_topics_top
import pandas as pd




data_dir = Path("data")
EXPERIMENT_OUTPUT_DIR = "output"

Test_Data_Dir = data_dir.joinpath("transcripts")
output_dir = data_dir.joinpath(EXPERIMENT_OUTPUT_DIR)
summary_output_dir = output_dir.joinpath("summary")
keyword_output_dir = output_dir.joinpath("keywords")
topic_output_dir = output_dir.joinpath("topic")

files = [fpath for fpath in Test_Data_Dir.iterdir() if fpath.exists() and fpath.is_file()]
file_path = st.selectbox("Select transcript file",files)
st.header(file_path.stem)
transcript_txt_lines = read_transcript(file_path)

output_list = []
for line in transcript_txt_lines:
        line_list = sent_tokenize(line)
        output_list.extend(line_list)
transcript_txt = "\n".join(output_list)

#https://stackoverflow.com/questions/17740833/checking-fuzzy-approximate-substring-existing-in-a-longer-string-in-python

# large_string = "thelargemanhatanproject is a great project in themanhattincity"
# query_string = "manhattan"

# def fuzzy_extract(qs, ls, threshold):
#     '''fuzzy matches 'qs' in 'ls' and returns list of 
#     tuples of (word,index)
#     '''
#     for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
#         print('word {}'.format(word))
#         for match in find_near_matches(qs, word, max_l_dist=1):
#             match = word[match.start:match.end]
#             print('match {}'.format(match))
#             index = ls.find(match)
#             yield (match, index)
st.write(transcript_txt)
summary_outfile = summary_output_dir.joinpath(f"summary_{file_path.stem}.json")
summary = get_summary(summary_outfile,transcript_txt_lines) 
st.write("# Summary")
st.write(summary)
#https://github.com/tvst/st-annotated-text

keyword_outfile = keyword_output_dir.joinpath(f"keyword_{file_path.stem}.json")            
keywords, ner_keywords, noun_keywords, over_all_keywords = get_keyword(keyword_outfile,summary)
st.write("# Hashtags")
hashtag_print = ", ".join([f"#{kw}" for kw in keywords])
st.write(hashtag_print)

# keywords, ner_keywords
st.write("# Keywords")
ner_keywords = [nr.split("_")[0] for nr in ner_keywords]



#TODO: IMPORTANT: THIS NEEDS TO BE REFACTORED FROM HERE AND PUT IT IN THE KEYWORD EXTRACTOR LOGIC ITSELF AS ADDITIONAL NER KEYWORDS
#Start
doc = nlp(transcript_txt)
for ent in doc.ents:
        if ent.label_ in [ 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']:
                if ent.text not in ner_keywords:
                        ner_keywords.append(f"{ent.text}")
# ner_keywords = [nr.split("_")[0] for nr in ner_keywords]

over_all_keywords = [kw[0] for kw in over_all_keywords]
for nr in ner_keywords:
        if nr not in over_all_keywords:
                over_all_keywords.insert(0, nr)
                # over_all_keywords.append(nr)





ner_keywords_print = ", ".join(ner_keywords)
st.write(ner_keywords_print)

# st.write("## noun_keywords")
# st.write(noun_keywords)

st.write("## over_all_keywords")
over_all_keywords_print = ", ".join(over_all_keywords)
st.write(over_all_keywords_print)

st.write("### Noun Keywords = Overall keywords - NER - KeyBert")
st.write(set(set(over_all_keywords) - set(ner_keywords)) - set(keywords) )

st.write("# Topic")
topic_dict_all = get_topics(summary)
topic_dict = get_topics_top(topic_dict_all, summary)
df = pd.DataFrame(topic_dict, columns =['Topic', 'Value'])
df.set_index('Topic', inplace=True)
st.bar_chart(df)


topic_dict_gpt3 = get_topic_gpt3(summary_outfile.name)
df_gpt3 = pd.DataFrame(topic_dict_gpt3, columns =['Topic', 'Value'])
df_gpt3.set_index('Topic', inplace=True)
st.bar_chart(df_gpt3)

# st.write(df)



