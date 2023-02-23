import textblob
import subprocess

cmd = ['python3','-m','textblob.download_corpora lite']
subprocess.run(cmd)
print("Word Data Downloaded")