import subprocess

cmd = ['python','-m','textblob.download_corpora']
subprocess.run(cmd)
print("Word Data Downloaded")