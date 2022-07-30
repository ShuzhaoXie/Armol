import sys
sys.path.join('..')
from common import json_dump, json_load, path_join

from scapy.all import *
from scapy.layers.dns import DNS, DNSQR, DNSRR, DNSRROPT
# from util import *
import numpy as np
import json
import time
import os
import shutil
import glob

lag_dir = './results/aws_us2sg'

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


def check_right_packet(ip, srcip):
    if ip.src == srcip or ip.dst == srcip:
        return True
    else:
        return False


def get_response_indices(packets, dstip):
    '''
    391 means failed call
    :param packets:
    :param dstip:
    :return:
    '''
    res = []
    for i, packet in enumerate(packets):
        if packet.haslayer(TLS) and packet[IP].src == dstip and packet[TLS].type == 23 and len(packet) > 391:
            res.append(i)

    return res


def get_last_ack(packets, ind, ip):
    for i in range(ind - 1, -1, -1):
        x = packets[i]
        if x.haslayer(TCP) and x[IP].src == ip and str(x[TCP].flags) == 'A':
            return i, x.time


def get_end_upload_timestamp(packets, ind, srcip):
    """
    只用找到 src 为 srcip的包就可以
    :param packets:
    :param ind:
    :param srcip:
    :return:
    """
    for i in range(ind - 1, -1, -1):
        x = packets[i]
        if x.haslayer(TCP) and x[IP].src == srcip:
            return i, x.time


def get_ack_before_first_upload_timestamp(packets, ind, srcip):
    """
    :param packets:
    :param ind:
    :param srcip:
    :return:
    """
    first_upload = ind - 1
    for i in range(ind - 1, -1, -1):
        x = packets[i]
        if x.haslayer(TLS) and x[IP].src == srcip and x[TLS].type == 23:
            first_upload = i
            break

    # 同一port的上一个包
    for i in range(first_upload - 1, -1, -1):
        x = packets[i]
        if x.haslayer(TCP) and x[IP].src == srcip:
            return i, x.time


def get_max(a):
    res = -1
    for x in a:
        if x > res:
            res = x
    return res


def get_min(a):
    res = 10000000000000000000
    for x in a:
        if x < res:
            res = x
    return res


def parse_latency(raw_packets, src_ip, cap_name):
    """

    :param raw_packets:
    :param src_ip:
    :param cap_name:
    :return:
    """

    packets = raw_packets
    len_packets = len(packets)

    dst_ip = get_dst_ip(packets, src_ip)
    response_indices = get_response_indices(packets, dst_ip)

    print('len response_indices', len(response_indices))
    print(response_indices)
    ul = []
    rl = []
    il = []

    last_max = -1
    for ind in response_indices:
        response = packets[ind]
        i2, t2 = get_last_ack(packets, ind, dst_ip)
        i1, t1 = get_end_upload_timestamp(packets, i2, src_ip)
        i0, t0 = get_ack_before_first_upload_timestamp(packets, i1, src_ip)

        print('i0 t0', i0, t0)
        print('i1 t1', i1, t1)
        print('i2 t2', i2, t2)

        t3 = response.time

        print('i3 t3', ind, t3)

        # print('t1-t0, t2-t1, t3-t2', t1-t0, t2-t1, t3-t2)

        ul.append(float(t1-t0))
        rl.append(float(t2-t1))
        il.append(float(t3-t2))

        if last_max == -1:
            last_max = get_max([i0, i1, i2, ind])
        else:
            last_min = get_min([i0, i1, i2, ind])
            if last_min <= last_max:
                print('LAST 4 ERROR')
            last_max = get_max([i0, i1, i2, ind])

    print('len', list(map(len, [ul, rl, il])))
    lags = [ul, rl, il]

    file_path = path_join(lag_dir, '{}.json'.format(cap_name))
    json_dump(lags, file_path)

    return np.mean(ul), np.mean(rl), np.mean(il)


def main():
    cap_dir_path = '/home/yy/Documents/dds/data/aws/origin/pcap'

    load_layer('tls')

    files = glob.glob(path_join(cap_dir_path, '*.cap'))

    src_ip = '172.31.43.57'

    for file in files:
        print(file)
        raw_packets = rdpcap(file)
        cap_name = file.split('/')[-1].split('.')[0]
        upload, rtt, infer = parse_latency(raw_packets, src_ip, cap_name)
        print('average', upload, rtt, infer)


def test():
    load_layer('tls')
    src_ip = '172.31.24.164'
    file = '/home/yy/Documents/dds/data/aws/us2sg/pcap/20210508_120101.cap'
    raw_packets = rdpcap(file)
    cap_name = file.split('/')[-1].split('.')[0]
    upload, rtt, infer = parse_latency(raw_packets, src_ip, cap_name)
    print('average', upload, rtt, infer)


def readable():
    load_layer('tls')
    src_ip = '172.31.24.164'
    file = '/home/yy/Documents/dds/data/aws/us2sg/pcap/20210508_120101.cap'
    raw_packets = rdpcap(file)
    ans = []
    dst_ip = get_dst_ip(raw_packets, src_ip)
    for i, packet in enumerate(raw_packets):
        if packet.haslayer(IP):
            if packet.haslayer(TLS) and packet[IP].src == dst_ip and packet[IP].dst == src_ip and packet[TLS].type == 23:
                ans.append([i, packet[IP].src, packet[IP].dst, len(packet)])
            # else:
            #     f.write('{} {} {}\n'.format(i, packet[IP].src, packet[IP].dst))
    l_min = 1000000
    f = open('./readable_test.txt', 'w')
    for i, tp in enumerate(ans):
        l_min = min(l_min, tp[3])
        if i == 0:
            f.write('{} {} {} {} \n'.format(tp[0], tp[1], tp[2], tp[3]))
        else:
            f.write('{} {} {} {} {}\n'.format(tp[0], tp[1], tp[2], tp[3], tp[0] - ans[i-1][0]))
    f.close()
    print('min', l_min)


if __name__ == '__main__':
    test()