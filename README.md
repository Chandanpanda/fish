# fish - Seven steps to use my deep learning server (2 mins)

# Step 1:
ssh -i /home/chapanda/.ssh/aws-key-fast-ai.pem ubuntu@ec2-52-41-99-115.us-west-2.compute.amazonaws.com

# Step 2:
git clone https://github.com/chandanpanda/fish	
chmod u+x fish/*.*
./fish/install-gpu.sh

# Step 3
./fish/getFiles.sh

# Step 4:
tmux

Ctrl + b , %

# Step 5:
jupyter notebook

# Step 6:
Open URL in browser
ec2-XX-XX-XXX-XX.us-west-2.compute.amazonaws.com:8888

# Step 7 

rsync -avp --progress aws-p2:~/fish/models/*.*

git config --global user.name "user_name"
git add ...
git commit -m "..."
git push origin master
