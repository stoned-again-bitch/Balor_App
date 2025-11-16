import os
training_dirs = ["happy","sad","angry","disgust","neutral","surprise"]
play_dir_base_path = "music"
for d in training_dirs:
    os.makedirs(d,exist_ok=True)
    os.makedirs(os.path.join(play_dir_base_path, d),exist_ok=True)

os.makedirs("songs_to_classify",exist_ok=True)
    

