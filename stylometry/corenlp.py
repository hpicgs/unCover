import os
import subprocess
import sys
import threading
import time
from urllib import request
import urllib.error

from definitions import ROOT_DIR

def __corenlp_thread():
    proc = subprocess.run([os.path.join(ROOT_DIR, 'deployment', 'corenlp')], capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout.decode())
    # we ignore exit code 2 because we might be waiting for another process to
    # finish the download
    if proc.returncode == 1:
        sys.exit(1)

# check if corenlp server is running, if not (download &) run it and wait until
# ready
def connect_corenlp(port = 9000):
    try:
        request.urlopen(f"http://localhost:{port}").getcode()
    except urllib.error.URLError:
        print("CoreNLP is not running. If you have not yet downloaded it, we will try to download it now before running it. This may take a few minutes as the model is about 500MB.")
        thread = threading.Thread(target=__corenlp_thread)
        thread.start()
        while True:
            try:
                request.urlopen(f"http://localhost:{port}").getcode()
                print("Success.")
                return
            except urllib.error.URLError:
                time.sleep(1)
