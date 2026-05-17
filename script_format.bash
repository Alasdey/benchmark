export HF_HOME="/home/jovyan/buckets"

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

uv run dataprep_llm_format.py --dataset Nofing/EventStoryLine-1.5-span
uv run dataprep_llm_format.py --dataset Nofing/Hievents-span
uv run dataprep_llm_format.py --dataset Nofing/MECI-v0.1-public-span
uv run dataprep_llm_format.py --dataset Nofing/Maven-ERE-span
uv run dataprep_llm_format.py --dataset Nofing/CausalTimeBank-aligned