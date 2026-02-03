export HF_HOME="/home/jovyan/buckets"

git clone https://github.com/nlp-uoregon/meci-dataset.git data/MECI

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

python3 MECI_dataprep.py \
  --root_dir data/MECI/meci-v0.1-public \
  --exclude 1_10ecbplus.xml \
  --seed 42 \
  --test_size 0.1 \
  --repo_id Nofing/MECI-v0.1-public-span
