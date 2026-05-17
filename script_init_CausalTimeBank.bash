export HF_HOME="/home/jovyan/buckets"

git clone https://github.com/paramitamirza/Causal-TimeBank.git data/CausalTimeBank
unzip data/CausalTimeBank/Causal-TimeBank-TimeML.zip -d data/CausalTimeBank/TimeML/

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

uv run python3 CausalTimeBank_dataprep_aligned.py \
  --root_dir data/CausalTimeBank/TimeML \
  --repo_id Nofing/CausalTimeBank-aligned
