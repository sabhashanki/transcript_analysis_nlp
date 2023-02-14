from pathlib import Path
from utils import read_transcript
import json

from summarizer import generate_summary
from keyword_extractor import get_keywords_from_summary, score_keywords


def get_summary(summary_outfile, transcript_txt_lines, is_force = False):
    if summary_outfile.exists() and (not is_force):
        with open(summary_outfile) as fp:
            summary_json = json.load(fp)
            summary = summary_json["summary"]
    else:
        summary = generate_summary(transcript_txt_lines)
        with open(summary_outfile,"w") as fp:
            json.dump({"summary":summary}, fp)
    return summary


def get_keyword(keyword_outfile,summary, is_force = False):
    if keyword_outfile.exists() and (not is_force):
        with open(keyword_outfile) as fp:
            keyword_content = json.load(fp)
            keywords = keyword_content["keywords"]
            ner_keywords = keyword_content["ner_keywords"]
            noun_keywords = keyword_content["noun_keywords"]
            over_all_keywords = keyword_content["over_all_keywords"]
            
    else:
        keywords, ner_keywords, noun_keywords = get_keywords_from_summary(summary)
        over_all_keywords = list(set(keywords + [nr.split('_')[0] for nr in ner_keywords] + noun_keywords))
        over_all_keywords = score_keywords(summary, over_all_keywords)
        with open(keyword_outfile,"w") as fp:
            json.dump({
                "keywords":keywords,
                "ner_keywords":ner_keywords,
                "noun_keywords":noun_keywords,
                "over_all_keywords":over_all_keywords
                },fp)
    return keywords, ner_keywords, noun_keywords, over_all_keywords

def create_output_directories(data_dir,output_dir_name,experiment_summary ):
    output_dir = data_dir.joinpath(output_dir_name)
    summary_output_dir = output_dir.joinpath("summary")
    keyword_output_dir = output_dir.joinpath("keywords")
    topic_output_dir = output_dir.joinpath("topic")
    output_dir.mkdir(exist_ok=True,parents=True)
    summary_output_dir.mkdir(exist_ok=True,parents=True)
    keyword_output_dir.mkdir(exist_ok=True,parents=True)
    topic_output_dir.mkdir(exist_ok=True,parents=True)
    with open(output_dir.joinpath("experiment_description.md"),"w") as fp:
        fp.write(experiment_summary)
    return summary_output_dir, keyword_output_dir, topic_output_dir




if __name__ == "__main__":
    EXPERIMENT_OUTPUT_DIR = "output"
    EXPERIMENT_DESCRIPTION = """
    Keep details of experiment here in markdown format...
    """
    data_dir = Path("data")
    Test_Data_Dir = data_dir.joinpath("transcripts")
    summary_output_dir, keyword_output_dir, topic_output_dir = create_output_directories(data_dir,output_dir_name=EXPERIMENT_OUTPUT_DIR, experiment_summary = EXPERIMENT_DESCRIPTION)
    idx = 0
    for fpath in Test_Data_Dir.iterdir():
        if fpath.is_file():
            idx = idx + 1
            print(f"{idx}: processing... File: {fpath.stem}")
            print("."*50)
            transcript_txt_lines = read_transcript(fpath)
            print(f"Number of sentences: {len(transcript_txt_lines)}")
            summary_outfile = summary_output_dir.joinpath(f"summary_{fpath.stem}.json")
            summary = get_summary(summary_outfile,transcript_txt_lines)
            print(f"Summary (word count : {len(summary.split())}): {summary}")
            print("."*50)
            keyword_outfile = keyword_output_dir.joinpath(f"keyword_{fpath.stem}.json")            
            keywords, ner_keywords, noun_keywords, over_all_keywords = get_keyword(keyword_outfile,summary)
            print(f"KeyBert Keywords (Count: {len(keywords)}): {keywords}")
            print(f"    NER Keywords (Count: {len(ner_keywords)}): {ner_keywords}")
            print(f"   Noun Keywords (Count: {len(noun_keywords)}): {noun_keywords}")
            print(f"OVERALL Keywords (Count: {len(over_all_keywords)}): {over_all_keywords}")
            print("_"*100)
  