export HF_HOME="/home/jovyan/buckets"

git clone https://github.com/paramitamirza/Causal-TimeBank.git data/CausalTimeBank
unzip data/CausalTimeBank/Causal-TimeBank-TimeML.zip -d data/CausalTimeBank/TimeML/

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

python3 CausalTimeBank_dataprep.py \
  --root_dir data/CausalTimeBank/TimeML \
  --seed 42 \
  --test_size 0.1 \
  --repo_id Nofing/CausalTimeBank-span
