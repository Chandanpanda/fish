#connect		#1
#ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-35-162-100-200.us-west-2.compute.amazonaws.com

#setup.sh		#2
git clone https://github.com/chandanpanda/fish
cd fish
chmod u+x install-gpu.sh
./install-gpu.sh

#enter password

cd ..
sudo apt install python-pip

#enter Y

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

jupyter notebook