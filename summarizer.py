import torch
from transformers import pipeline
from utils import preprocess_transcript, divide_chunks

num_of_gpus = torch.cuda.device_count()
# TO know how many GPUs are available
print(num_of_gpus)

if num_of_gpus:
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
else:
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")




def summarize_chunks(transcript_txt_chunks):
    sum = []
    for chunk_lines in transcript_txt_chunks:
        # print(f"TEXT CHUNK: {' '.join(chunk_lines)}")
        chunk_text = ' '.join(chunk_lines)
        MAX_LENGTH = len(chunk_text.split())//10 + 1  #10%
        MIN_LENGTH = len(chunk_text.split())//50 + 1  #2%
        try:
            chunk_sum = summarizer(chunk_text, max_length = MAX_LENGTH, min_length = MIN_LENGTH)
            sum.append(chunk_sum[0].get('summary_text'))
            # print(f"CHUNK SUMMARY: {sum[-1]}")
        except:
            pass
    summary_text = ' '.join(sum)
    return summary_text

def generate_summary(transcript_txt_lines):
    assert(type(transcript_txt_lines)==list)
    #TODO: change the num_word_th=0 and don't filter anything. It might give better summary
    #However, for this, change the divide chunk logic and now chunk based on tokens or words rather than sentences
    transcript_txt_lines = preprocess_transcript(transcript_txt_lines, num_word_th=10)
    transcript_txt_chunks = divide_chunks(transcript_txt_lines)
    summary = summarize_chunks(transcript_txt_chunks)
    return summary

if __name__ == "__main__":
    from pathlib import Path
    # from summarizer import Summary
    from utils import remove_timestamp
    
    Test_Data_Dir = Path("data/transcripts")
    for fpath in Test_Data_Dir.iterdir():
        if fpath.is_file():
            print(f"processing {fpath.stem}")
            print("."*50)
            with open(fpath,"r") as fp:
                transcript_txt = fp.read()
            transcript_txt_lines = transcript_txt.split("\n")
            #cleanup lines which are timestamps or number of words are less than 5-6.
            transcript_txt_lines = remove_timestamp(transcript_txt_lines)
            print(f"Number of sentences: {len(transcript_txt_lines)}")
            summary = generate_summary(transcript_txt_lines)
            print(f"Summary: {summary}")
            print(f"Summary Length: {len(summary.split())}")
            print("_"*100)