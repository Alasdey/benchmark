export HF_HOME="/home/jovyan/buckets"

git clone https://github.com/tommasoc80/EventStoryLine.git data/EventStoryLine

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

python3 EventStoryLine_dataprep.py \
  --root_dir data/EventStoryLine/annotated_data/v1.5 \
  --exclude 1_10ecbplus.xml \
  --seed 42 \
  --test_size 0.1 \
  --repo_id Nofing/EventStoryLine-1.5-span