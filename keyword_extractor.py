from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity

from utils import  divide_chunks
from nltk.tokenize import sent_tokenize

from spacy.symbols import nsubj,nsubjpass,dobj,iobj,pobj


kw_extractor = KeyBERT('sentence-transformers/all-mpnet-base-v2')
nlp = spacy.load('en_core_web_md')



def predict(summary_text):
    keybert_diversity_phrases = []
    NER_Keywords = []
    noun_keywords = []
    for new_text in summary_text:
        try:
            keywords_n = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.*>'), use_mmr=True, diversity=1.0,
                                                        keyphrase_ngram_range=(1, 1), stop_words='english', top_n=50)
            # print(keywords_n)
            keywords_noun = [i for i in keywords_n if i[1] > 0.2]
            for i, _ in keywords_noun:
                keybert_diversity_phrases.append(i)
        except:
            print('no keywords found for this loop-a')
        try:
            keywords2_nn = kw_extractor.extract_keywords(new_text, vectorizer=KeyphraseCountVectorizer(pos_pattern='<N.+>+<N.+>'), use_mmr=True, diversity=1.0,
                                                            keyphrase_ngram_range=(2, 3), stop_words='english', top_n=50)
            keywords_nnounn = [i for i in keywords2_nn if i[1] > 0.2]
            for i, _ in keywords_nnounn:
                keybert_diversity_phrases.append(i)
        except:
            print('no keywords found for this loop-b')
        #extract ner keywords
        doc = nlp(new_text)
        for noun_chunk in doc.noun_chunks:
            # text = str(noun_chunk)
            # regex = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"
            # text = re.sub(regex, "", text)
            # text = text.strip()
            # if text.isdigit():        
            #     print(f"filtering {noun_chunk} as it doesn't contain any character")
            #     continue
            if noun_chunk.root.pos_ in ["PROPN", "NOUN"]:
                if not any(word.is_stop for word in noun_chunk):
                    #TODO: apply some basic filtering, such as pronpouns and other stop word removal.
                    noun_keywords.append(noun_chunk.text)
                elif (not any(word.is_stop for word in noun_chunk[1:])) and noun_chunk.text.split(" ")[0].lower() == "the":
                    noun_keywords.append(noun_chunk.text) 
                else:
                    print(f"filtered any stop words: {noun_chunk.text}")
                
            else:
                print(f"filtering {noun_chunk.text}")
        
        for ent in doc.ents:
            if ent.label_ in [ 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']:
                NER_Keywords.append(f"{ent.text}_({ent.label_.lower()})")
        
        #Experiments starts
        # print(f"EXPERIMENTAL")
        # for item in iter_nps(new_text):
        #     print(item)
        # #experiment ends
        
    keybert_diversity_phrases = list(set([i.lower() for i in keybert_diversity_phrases]))
    NER_Keywords = list(set([i.lower() for i in NER_Keywords]))
    noun_keywords = list(set([i.lower() for i in noun_keywords]))

    return keybert_diversity_phrases, NER_Keywords, noun_keywords

def score_keywords(summary:str,keywords:list[str]):
    keyword_score_list_sorted = []
    if len(keywords) > 0:
        sen_encode = kw_extractor.model.embed(summary)
        keyword_encode = kw_extractor.model.embed(keywords)
        score = cosine_similarity(sen_encode.reshape(1, -1), keyword_encode)
        score = [int(s*1000) for s in score[0]]
        #to save score in json file, it shouldn't be float and hence converting it to int
        # print(f"Score[0]={score[0]} and type: {type(score[0])}")
        keyword_score_list_sorted = sorted(zip(keywords, score), key = lambda x : x[1], reverse=True)
    return keyword_score_list_sorted#keyword_score_list_sorted

def get_keywords_from_summary(summary):
    summary_line_list = sent_tokenize(summary)
    summary_line_chunks_list = divide_chunks(summary_line_list)
    summary_chunks_txt_list = [' '.join(i) for i in summary_line_chunks_list]
    keywords, ner_keywords, noun_keywords = predict(summary_chunks_txt_list)
    return  keywords, ner_keywords, noun_keywords