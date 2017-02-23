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
Enter Y

# Step 4:
./start.sh

# Step 5:
Ctrl + b , %

# Step 6:
jupyter notebook
