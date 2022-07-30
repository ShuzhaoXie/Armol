# I rent some AWS virtual machines to send the requests, here are the commands that we use config the fresh VMs.
sudo add-apt-repository universe
sudo apt-get update
sudo apt install python3-pip
sudo pip3 install absl-py
sudo pip3 install --upgrade azure-cognitiveservices-vision-computervision
sudo pip3 install pillow