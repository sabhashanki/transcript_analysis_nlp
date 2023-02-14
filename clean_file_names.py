from pathlib import Path
import re
import unicodedata

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

DATA_DIR = Path("data/output")
transcript_dir = DATA_DIR.joinpath("summary")
for fpath in transcript_dir.iterdir():
    print(fpath)
    print(f"New Name:{slugify(fpath.stem)}.json")
    fpath.rename(transcript_dir.joinpath(f"{slugify(fpath.stem)}.json"))
    # fpath.rename(transcript_dir.joinpath(f"{fpath.stem[8:]}.json"))