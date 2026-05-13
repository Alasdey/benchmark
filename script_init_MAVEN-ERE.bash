# mkdir -p data/MAVEN-ERE

# wget -O data/MAVEN-ERE/MAVEN_ERE.zip https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1
# unzip data/MAVEN-ERE/MAVEN_ERE.zip -d data/MAVEN-ERE/

# wget -O data/MAVEN-ERE/processed.zip https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1
# unzip data/MAVEN-ERE/processed.zip -d data/MAVEN-ERE/

read -s -r -p "Enter your Hugging Face token: " HF_TOKEN
export HF_TOKEN

python3 Maven-ERE_dataprep.py