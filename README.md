# fish - Seven steps to use my deep learning server (20 mins)

# Step 1:
ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-52-41-99-115.us-west-2.compute.amazonaws.com

# Step 2:
git clone https://github.com/chandanpanda/fish	

# Step 3

pip install kaggle-cli

sudo apt-get install unzip

kg config -g -u "ChandanPanda2006" -p "Passw0rd" -c "the-nature-conservancy-fisheries-monitoring"

kg download

# Step 4 

unzip train.zip  

mkdir test

unzip test_stg1.zip -d "test"

mkdir results

mkdir valid

rm -rf train/__MACOSX

rm train/.DS_Store

rm -rf test/test_stg1/__MACOSX


# Step 5:
tmux

Ctrl + b , %


# Step 6:
jupyter notebook

# Step 7:
Open URL in browser
ec2-XX-XX-XXX-XX.us-west-2.compute.amazonaws.com:8888

git add ...
git commit -m "..."
git push origin master
