# fish - Seven steps to use my deep learning server (2 mins)

# Step 1:
ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-52-41-99-115.us-west-2.compute.amazonaws.com

# Step 2:
git clone https://github.com/chandanpanda/fish	
chmod u+x fish/*.*

# Step 3
pip install kaggle-cli

./fish/install-gpu.sh

reboot

sudo apt-get install unzip

kg config -g -u "ChandanPanda2006" -p "Passw0rd" -c "the-nature-conservancy-fisheries-monitoring"

kg download

unzip train.zip

mkdir test

unzip test_stg1.zip -d "test"

mkdir results

mkdir valid

rm -rf train/__MACOSX

rm train/.DS_Store

rm -rf test/test_stg1/__MACOSX

conda install opencv

# Step 4:
tmux

Ctrl + b , %

# Step 5:
jupyter notebook

# Step 6:
Open URL in browser
ec2-XX-XX-XXX-XX.us-west-2.compute.amazonaws.com:8888

# Step 7 

#scp -i aws_eu.pem unet_42quality.hdf5  ubuntu@ec2-34-248-60-241.eu-west-1.compute.amazonaws.com:~/fish/
#scp -i aws_eu.pem ubuntu@ec2-34-248-60-241.eu-west-1.compute.amazonaws.com:~/fish/filename filename
