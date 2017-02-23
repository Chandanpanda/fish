# fish - Seven steps to use my deep learning server

# Step 1:
ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-35-162-100-200.us-west-2.compute.amazonaws.com

# Step 2:
git clone https://github.com/chandanpanda/fish	

# Step 3:
chmod u+x fish/*.*
cd fish
./install-gpu.sh
Enter password

# Step 4:
sudo apt install python-pip

Enter Y

pip install kaggle-cli

sudo apt-get install unzip

kg config -g -u "ChandanPanda2006" -p "Passw0rd" -c "the-nature-conservancy-fisheries-monitoring"

kg download

mkdir train

unzip train.zip  

mkdir test

unzip test_stg1.zip 

mv test_stg1 test 

mkdir results

mkdir valid

rm -rf train/__MACOSX

rm train/.DS_Store

rm -rf test/__MACOSX

rm test/.DS_Store


# Step 5:
tmux

Ctrl + b , %


# Step 6:
jupyter notebook

# Step 7:
Open URL in browser
ec2-XX-XX-XXX-XX.us-west-2.compute.amazonaws.com:8888

