# fish - Seven steps to use my deep learning server (2 mins)

# Step 1:
ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-52-41-99-115.us-west-2.compute.amazonaws.com

# Step 2:
git clone https://github.com/chandanpanda/fish	
chmod u+x fish/*.*

# Step 3
pip install kaggle-cli

reboot

sudo dpkg --configure -a

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

# Step 4:
tmux

Ctrl + b , %

# Step 5:
jupyter notebook

# Step 6:
Open URL in browser
ec2-XX-XX-XXX-XX.us-west-2.compute.amazonaws.com:8888

# Step 7 

zip myData.zip "file1" ...
download file
