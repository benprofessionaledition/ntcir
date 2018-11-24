#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:41:57 2018

@author: chungchi <--- you suck at coding, "chungchi" <3 Ben
"""

import logging
import time

logging.basicConfig(
    format='%(asctime)s %(levelname)-s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# use browser to get code
# code_url = "https://api.stocktwits.com/api/2/oauth/authorize?client_id=" + consumer_key + "&response_type=code&redirect_uri=" + redirect_url + "&scope=read,watch_lists,publish_messages,publish_watch_lists,direct_messages,follow_users,follow_stocks"

import json
import os

import requests

consumer_key = ["e25d9d096974cd7d", "339361c6bdf0cb4e",
                "5729188159519227", "011b0f7fc08b78a0",
                "1f2a2422efe84ad9", "a350d76979a3c964",
                "f9a4747ec481a5df", "3f37cca6691d1cb7",
                "090896c503c5ae0b", "20220ba629d9d3bd",
                "ee9956eede6248f6", "4b52b3982d7d39c1",
                "71f24ea4af374798", "b7282bf258e0c264",
                "ecb39baa2b75c5ec"]

consumer_secret = ["a667a3968e7497f150207d265ca26a611821e516", "a9eac5f14adc8e70a4a7708ccf8db150ebf0d6cd",
                   "0881531d5f071cb0175448943a44f6699c275b74", "d863d7cae370291a46f2809e0b01370c0b211e26",
                   "4a96dd8ff2f744ad7a452d3054f604a4105280ea", "3399938751bd1c26a18d223f23e54c7c44b48bea",
                   "53c9e7c0d11963621b614290be15e4a1b58dfd3b", "472e6e9cfe33fbf3625ecf09a1144921ca51120b",
                   "3fd676b62bf195037a79b7d40e9a94fab62e14e9", "840714a5446ce00bd1aa12d41e2826325ab40df4",
                   "1d7f37a5ae5698038dc86cadc2b914bcb89b1505", "7bfeaa82c987fd06c7c60777a61f83f9c70d6a77",
                   "69e08721b5e220163ab3050a174e57c0cdd0a6e2", "f774c041f3dcbb74dd2659ad122be81cb3f71b76",
                   "e549c243c4aa37cbc5e5060b71781b06f74453c8"]

redirect_url = "http://www.google.com"

# these expire apparently so you need new ones
code = ["16182a66fb98dd604b2d04736b8b1b54e5f9ff1a", "7a168744513614932c42fb8f6e43cff6bdced763",
        "a6d079739e9e9b4db3e49a4b655a318e5f667e65", "a380cf89f20f6f3209440c712db1cdbff58d7a27",
        "9b94c23324808f38d97b92ed5d876b9871b4b4f8", "a10ae0bf0b530352f8fe86c4bb5cfe963ba2dfc8",
        "f16da79451095243c3a3508350c43df5e8bffb82", "8d6123a81e2849845122b8a54bac9e1271993e11",
        "9e7e6a6756aa0d1388fb627768993610a0b7dac8", "6149c82ac339bc71d65db2856424d064a054f8cb",
        "3e66a35006994c5bc5777b31806148e4823ccda8", "dd297f2dd9d504d6e63c79f932aa889d17c66f66",
        "8df66881b101212135f339fb42b46851f3fa9c2b", "7e622c0411890060594c8c7c6486b1f1fc02bc21",
        "258a3d26bc8bac6ec99e71c46e45bef65a248467"]


def next_token(j: int) -> str:
    token_url = ["https://api.stocktwits.com/api/2/oauth/token?client_id=" \
                 + consumer_key[i] + "&client_secret=" + consumer_secret[i] + "&code=" \
                 + code[i] + "&grant_type=authorization_code&redirect_uri=" + redirect_url for i in range(len(consumer_key))]
    token_info = requests.post(token_url[j])
    try:
        return json.loads(token_info.content.decode("utf8"))["access_token"]
    except:
        print("Couldn't parse response: {}".format(token_info))
        if j < len(consumer_key) - 1:
            return next_token(j + 1)
        else:
            raise ValueError("all tokens exhausted")


if __name__ == '__main__':
    j = 0
    token = next_token(j)
    for fname in ["FinNum_training", "FinNum_dev"]:
        with open(os.path.join("/Users/ben.levine/ntcir/resources/ignored/", fname + ".json")) as f:
            data = json.load(f)

        # Authenticated calls are permitted 400 requests per hour and measured against the access token used in the request.
        not_found = []
        twt = []
        idx = []

        i = 4072
        while i != len(data):
            print(i)
            if data[i]["idx"] in idx:
                j = idx.index(data[i]["idx"])
                data[i]["tweet"] = twt[j]
                i = i + 1
                continue

            url = "https://api.stocktwits.com/api/2/messages/show/" + str(data[i]["id"]) + ".json?access_token=" + token
            tweet_info = json.loads(requests.get(url).content.decode("utf8"))

            if tweet_info["response"]["status"] == 200:
                tweet = tweet_info["message"]["body"]
                data[i]["tweet"] = tweet
                twt.append(tweet)
                idx.append(data[i]["idx"])
                i = i + 1
            elif tweet_info["response"]["status"] == 429:
                print("next url...")
                j += 1
                chkpt = fname + "_rebuilt.json.ckpt-" + str(time.time())
                log.warning("token exhausted, dumping checkpoint to %s", chkpt)
                json.dump(data, open(chkpt, 'w'))
                token = next_token(j)
            else:
                not_found.append(i)
                print(i)
                print(tweet_info)
                i = i + 1

        for i in not_found[::-1]:
            del data[i]

        print("Missing data: " + str(len(not_found)))
        print("Total: " + str(len(data)) + " instances")

        json.dump(data, open(fname + "_rebuilt.json", 'w'))
