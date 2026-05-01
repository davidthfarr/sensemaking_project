import json
import os

# Get the current working directory
cwd = os.getcwd()
env = {}
f = open(cwd + "/environment.json")

try:
    env = json.loads(f.read())
except:
    raise Exception("no environment found")

def RAW_FILE_PATH():
    return env.get('raw_filepath')

def CLEANED_FILE_PATH():
    return env.get('cleaned_filepath')

def PROCESSED_FILE_PATH():
    return env.get('processed_filepath')

def EVALUATED_DIR():
    return env.get('evaluated_dir')

def OUTPUT_HTML_FILE_PATH():
    return env.get('output_html_filepath')
