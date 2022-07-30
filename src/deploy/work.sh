time=$(date "+%Y%m%d_%H%M%S")
tcpfilename="${time}.cap"
echo $tcpfilename

sudo tcpdump -U -i eth0 -w $tcpfilename &
python3 /home/ec2-user/aws_od_sender.py --location=hongkong --region=hk-sg

pid=$(ps -e | pgrep tcpdump)
echo $pid

sleep 2
sudo kill -2 $pid