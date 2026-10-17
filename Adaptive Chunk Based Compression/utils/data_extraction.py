import zipfile

def extract_newsgroups():
    ZIP_PATH = './data/6_newsgroups.zip'
    OUT_DIR = './data'
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(OUT_DIR)

    print(f"Extracted {ZIP_PATH} to {OUT_DIR}")