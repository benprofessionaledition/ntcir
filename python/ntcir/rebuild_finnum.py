#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:41:57 2018

@author: chungchi <--- you suck at coding, "chungchi" <3 Ben
"""

#use browser to get code
#code_url = "https://api.stocktwits.com/api/2/oauth/authorize?client_id=" + consumer_key + "&response_type=code&redirect_uri=" + redirect_url + "&scope=read,watch_lists,publish_messages,publish_watch_lists,direct_messages,follow_users,follow_stocks"

import json
import time
import requests

consumer_key = ["e25d9d096974cd7d", "339361c6bdf0cb4e",
                "5729188159519227", "011b0f7fc08b78a0"]

consumer_secret = ["a667a3968e7497f150207d265ca26a611821e516", "a9eac5f14adc8e70a4a7708ccf8db150ebf0d6cd",
                   "0881531d5f071cb0175448943a44f6699c275b74", "d863d7cae370291a46f2809e0b01370c0b211e26"]

redirect_url = "http://www.google.com"

code = ["6ee651e5a1d692f6e00d86d44c7d2da51d75cfa7", "27e042cd91d437b1a55ff952153ea4b55d7c324e",
        "9c9159773a137382ea2d1934b1b2cd030049b913", "f6594051d0be9514fe5b24c57c4839afcbf3fa67"]

token_url = "https://api.stocktwits.com/api/2/oauth/token?client_id=" + consumer_key + "&client_secret=" + consumer_secret + "&code=" + code + "&grant_type=authorization_code&redirect_uri=" + redirect_url


token_info = requests.post(token_url)
token = json.loads(token_info.content.decode("utf8"))["access_token"]

for FinNum in ["FinNum_training","FinNum_dev"]:
    with open(FinNum + ".json") as f:
        data = json.load(f)
    
    #Authenticated calls are permitted 400 requests per hour and measured against the access token used in the request.
    not_found = []
    twt = []
    idx = []
    
    i = 0
    while(i != len(data)):
        print(i)
        if(data[i]["idx"] in idx):
            j = idx.index(data[i]["idx"])
            data[i]["tweet"] = twt[j]
            i = i + 1
            continue
        
        url = "https://api.stocktwits.com/api/2/messages/show/" + str(data[i]["id"]) + ".json?access_token=" + token
        tweet_info = json.loads(requests.get(url).content.decode("utf8"))
    
        if(tweet_info["response"]["status"] == 200):
            tweet = tweet_info["message"]["body"]
            data[i]["tweet"] = tweet
            twt.append(tweet)
            idx.append(data[i]["idx"])
            i = i + 1
        elif(tweet_info["response"]["status"] == 429):
            print("sleep one hour----from " + time.ctime() )
            time.sleep(3600)
        else:
            not_found.append(i)
            print(i)
            print(tweet_info)
            i = i + 1
            
    for i in not_found[::-1]:
        del data[i]
    
    print("Missing data: " + str(len(not_found)))
    print("Total: " + str(len(data)) + " instances")
    
    json.dump(data, open(FinNum + "_rebuilt.json", 'w'))