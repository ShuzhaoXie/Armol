import json
import sys
sys.path.append('..')
from common import path_join, parse_day_hour
from scapy.all import *
import time
import io
from scapy.main import load_layer
from scapy.utils import rdpcap

import glob



def get_dst_ip(pkts, srcip):
    # for pkt in pkts:
    #     if pkt.haslayer(DNS):
    #         dns = pkt[DNS]
    #         if dns.qr == 0:
    #             continue
    #         else:
    #             qd = dns.qd
    #             if qd.qtype == 1:
    #                 an = dns.an
    #                 ip = an.rdata
    #                 return ip
    mp = {}
    for pkt in pkts:
        if pkt.haslayer(IP):
            ip = pkt[IP]
            if ip.src == srcip:
                dstip = ip.dst
                if dstip in mp:
                    mp[dstip] += 1
                else:
                    mp[dstip] = 1

    dip = max(mp, key=mp.get)
    return dip


def count_key_cap(path):
    raw_packets = rdpcap(path)
    src_ip = '10.2.0.4'
    dst_ip = get_dst_ip(raw_packets, src_ip)
    cnt = 0

    for i, packet in enumerate(raw_packets):
        if packet.haslayer(TLS) and packet[IP].src == dst_ip and packet[IP].dst == src_ip and packet[TLS].type == 23 and len(packet) == 116:
            cnt += 1

    print(cnt)

    return cnt, dst_ip


def main(DAY):
    cap_dir_path = '/home/yy/Documents/dds/data/google/sg2sg{}/pcap'.format(DAY)

    load_layer('tls')

    files = glob.glob(path_join(cap_dir_path, '*.cap'))

    print(files)
    src_ip = '10.2.0.4'

    sum_cnt = 0
    dst_ips = ['' for _ in range(24)]
    for file in files:
        print(file)
        day, hour = parse_day_hour(file.split('/')[-1].split('.')[0])
        cnt, dst_ip = count_key_cap(file)
        sum_cnt += cnt
        dst_ips[int(hour)] = dst_ip
        # raw_packets = rdpcap(file)
        # cap_name = file.split('/')[-1].split('.')[0]
        # upload, rtt, infer = parse_latency(raw_packets, src_ip, cap_name)
        # print('average', upload, rtt, infer)

    # with open('test_dst_ip.txt', 'w') as f:
    #     for ip in dst_ips:
    #         f.write(ip + '\n')

    return dst_ips


if __name__ == '__main__':
    res = []
    for day in [16, 17, 18]:
        res.extend(main(day))

    with open('test_dst_ip.txt', 'w') as f:
        for ip in res:
            f.write(ip + '\n')