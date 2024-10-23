'''
This contains some utilities for processing 
network traffic. 

The constant definitions are also here. 

'''
import pandas as pd 
import pytz
from datetime import datetime
import utils as ut

TS = 'ts'
DURATION = 'duration'
SRC_IP = 'id.orig_h'
SRC_PORT = 'id.orig_p'
DEST_IP = 'id.resp_h'
DEST_PORT = 'id.resp_p'


#CONNECTION LOG FIELDS 
PROTO_FIELD = 'proto'
BYTES_OUT = 'orig_ip_bytes'
PKTS_OUT = 'orig_pkts'
BYTES_IN = 'resp_ip_bytes'
PKTS_IN = 'resp_pkts'
SERVICE_FIELD = 'service'
CONN_STATE = 'conn_state'
PAYLOAD_BYTES_OUT = 'orig_bytes'
PAYLOAD_BYTES_IN = 'resp_bytes'
SRC_MAC = 'orig_l2_addr'
DEST_MAC = 'resp_l2_addr'

CONN_NUMERIC_FEATURES = [BYTES_OUT,PKTS_OUT,BYTES_IN,PKTS_IN,PAYLOAD_BYTES_OUT,PAYLOAD_BYTES_IN]
CONN_CATEGORICAL_FEATURES = [PROTO_FIELD, DEST_IP, DEST_PORT]

#HTTP LOG FIELDS
HTTP_URI = 'uri'
HTTP_USER_AGENT = 'user_agent'
HTTP_HOST = 'host'
HTTP_RSP_SIZE = 'response_body_len'
HTTP_STATUS = 'status_code'

HTTP_NUMERIC_FEATURES=[HTTP_RSP_SIZE]
HTTP_CATEGORICAL_FEATURES=[DEST_IP,HTTP_URI, HTTP_USER_AGENT,HTTP_HOST]

#SSL LOG FIELDS
SSL_SERVER = 'server_name'
SSL_SUBJECT = 'subject'
SSL_ISSUER = 'issuer'
SSL_CLIENT_SUBJECT = 'client_subject'
SSL_CLIENT_ISSUER = 'client_issuer'
 
#DNS LOG FIELDS
DNS_QUERY = 'query'
DNS_RESPONSE = 'answers'

DNS_NUMERIC_FEATURES=[]
DNS_CATEGORICAL_FEATURES=[DNS_QUERY]

#DHCP LOG FIELDS 
DHCP_CLIENT='client_addr'
DHCP_MAC='mac'
DHCP_SERVER='server_addr'
DHCP_ASSIGNED='assigned_addr'
DHCP_SUBNET_MASK='subnet_mask'
DHCP_ROUTER='routers'
DHCP_DNS='dns_servers'

'''
Find the unique subsets from named columns 
'''
def get_unique_subset(df, cols):
    entries = df[cols].drop_duplicates().dropna()
    return entries
'''
Find unique entries from a single column 
is_list marks if the column contains a list 
'''
def get_unique_entries(df, col, is_list=False):
    if is_list:
        this_series = list(df[col].dropna())
        new_list = [x for y in this_series for x in y]
        return list(set(new_list))
        
    else:
        entries = df[col].dropna().unique()
        return list(entries)

from ipaddress import IPv4Address, IPv6Address

def is_valid_ipv4(addr):
    try:
        ip = IPv4Address(addr)
        return True
    except:
        return False

def is_valid_ipv6(addr):
    try:
        ip = IPv6Address(addr)
        return True
    except:
        return False
    



def get_mapping_dict(df, ip_field, mac_field, skip_mac):
    this_map = get_unique_subset(df,[ip_field, mac_field])
    answer = dict()
    grouped = this_map.groupby(mac_field)
    for name, group in grouped: 
        if name != skip_mac:
            answer[name] = list(set(list(group[ip_field])))
    return answer    

def set2dict_ipaddr(this_list):
    #convert a set of ipaddrs to a dict where key is first ipv4 address 
    # every other address maps to this ipv4 address. 
    # Find the first entry in the list which is a ipv4 address 
    this_list = list(set(this_list))
    if len(this_list) == 1:
        return dict()
    ipv4 = None 
    for x in this_list:
        if is_valid_ipv4(x) and x not in ['0.0.0.0', '::']:
            ipv4=x
            break
    answer = dict()
    if ipv4 is None:
        # Let every entry in this list map to itself. 
        # There is no IPv4 equivalence for this MAC Address 
        return dict()
    else:
        for x in this_list: 
            if x not in ['0.0.0.0', '::'] and x != ipv4:
                answer[x] = ipv4
    return answer   

def find_client_mapping(df, skip_mac):
    #This will return a dictionary which maps IPV6 addresses to IPV4 addresses. 
    client_dict = get_mapping_dict(df, SRC_IP, SRC_MAC, skip_mac)
    server_dict = get_mapping_dict(df, DEST_IP, DEST_MAC, skip_mac)
    for key in client_dict.keys():
        if key in server_dict.keys():
            server_dict[key] = list(set(server_dict[key]+client_dict[key]))
        else:
            server_dict[key] = client_dict[key]
    #Now server_dict has all the IPV6 and IPV4 addresses that have same MAC
    #We will convert those to a new dict 
    answer_dict = dict()
    for key in server_dict.keys():
        this_dict = set2dict_ipaddr(server_dict[key]) 
        #check that none of the new keys are in answer_dict already 
        for key in this_dict.keys():
            if key in answer_dict.keys():
                print(f'Duplicate key {key}')
                continue
        answer_dict.update(this_dict)
    return answer_dict    

def conn_map_addresses(df, map_dict):
    def map_address(x):
        return map_dict.get(x,x)

    df[SRC_IP] = df[SRC_IP].apply(map_address)
    df[DEST_IP] = df[DEST_IP].apply(map_address)
    return df

def read_n_map(filename, map_dict):
    df = pd.read_json(filename, lines=True)
    return conn_map_addresses(df, map_dict)



def get_server_port(local_server, conn_df):
    subset=conn_df[conn_df[DEST_IP] == local_server]
    subset = subset[subset[CONN_STATE]=="SF"]
    unique_ports = list(set(list(subset[DEST_PORT])))
    return unique_ports

def find_clients(addr, conn_df):
    this_df = conn_df[conn_df[DEST_IP]==addr]
    this_df = this_df[this_df[CONN_STATE]=="SF"]
    return this_df[[SRC_IP, DEST_PORT]].drop_duplicates()

def add_hour(df, field_name='Hour', local_prefix=None):
    aus_tz = pytz.timezone('Australia/Sydney')
    def compute_hour(second_value):
        this_time = datetime.fromtimestamp(second_value,aus_tz)
        return this_time.hour

    df[field_name] = df[TS].apply(compute_hour)
    if local_prefix is not None:
        df = df[df[SRC_IP].str.startswith(local_prefix)]
    return df