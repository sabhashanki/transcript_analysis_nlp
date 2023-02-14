import pandas as pd
# import requests
from pathlib import Path
import urllib.request  # the lib that handles the url stuff
import re




if __name__ == "__main__":
  test_data_list = "data/STTs.csv"
  test_data_dir = Path("data/transcripts")
  test_data_dir.mkdir(exist_ok=True, parents=True)
  regex = re.compile('[^a-zA-Z]')
  df = pd.read_csv(test_data_list)
  for ind in df.index:
      ftitle = df["name"][ind].strip()
    #   print(ftitle)
      furl = df["stt_file_path"][ind].strip()
    #   print(furl)
      text_list = []
      for line in urllib.request.urlopen(furl):          
          line = line.strip()
          line = line.decode("utf-8")
          # if len(line) < 30:
          #   #   print(type(line.decode("utf-8")))
          #     tmp_line = regex.sub('', line)
          #     if len(tmp_line) > 2:
          #       text_list.append(line)                  
          # else:
          #     text_list.append(line)
          text_list.append(line)
      with open(test_data_dir.joinpath(f"{ftitle}.txt"),"w") as fp:
          fp.write("\n".join(text_list))

