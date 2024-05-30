import csv
import requests
import json
import random
import time

def init_requete(url, params):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15",
    }   
    reponse = requests.get(url, headers=headers, params=params)
    if reponse.status_code == 200:
        return reponse.text
    print("fail to connect")
    return -1

def random_sleep():
    sleep_time = random.uniform(0, 0.2)
    time.sleep(sleep_time)

def store_data(fic_name, title, raw_data):
    with open(fic_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        ls = process_string(title)
        csv_writer.writerow(process_string(title))
        for row in raw_data:
            csv_writer.writerow(process_string(row))

def transforme_time(time_string):
    # 将时间字符串中的特熟符号改写为ascii
    ls = time_string.split(":")
    return ("%3A").join(ls)
    
def combine_web(size, nbPage, inalizedAt):
    # 返回链接的参数设置
    return { "size": size,
        "page": nbPage,
        "q_mode": "complete",
        "truncate": "50",
        "inalizedAt": inalizedAt
    }

def process_string(ls):
    res = []
    for el in ls:
        el = el.replace("_", " ")
        res.append(f'{el}')
    return res

#if __name__=='__main__':
def main(page_name):
    host= "https://data.ademe.fr"
    #path = "/data-fair/api/v1/datasets/agribalyse-31-detail-par-ingredient"
    #path = "/data-fair/api/v1/datasets/agribalyse-31-detail-par-etape"
    #path = "/data-fair/api/v1/datasets/agribalyse-31-synthese"
    path = "/data-fair/api/v1/datasets/" + page_name
    title_res = init_requete(host+path, {"html" : "true"})
    # 初始化基本信息, 用来组合后一步链接
    json_data = json.loads(title_res)
    total_lines = json_data["count"]
    inalized_time = json_data["finalizedAt"]
    # 获取表头内容
    form_title = {}
    for el in json_data["schema"]:
        form_title.update({el["key"]:el["description"]})
    
    # 开始读取数据内容
    url_line = host+path+"/lines"
    res = []    # 存放每一行的数据，List[List]
    size = 20
    for nbPage in range(1, total_lines//size + 1):
        print("Start crabing page :" + str(nbPage))
        line_res = init_requete(url_line, combine_web(size, nbPage, transforme_time(inalized_time)))
        json_data = json.loads(line_res)
        for line in json_data["results"]:
            row = [] # 存放一行数据
            for key in form_title:
                try:
                    row.append(str(line[key]))
                except KeyError as e:
                    row.append(str(""))
                
            res.append(row)
        random_sleep()
    print("finish crabing")
    store_data(page_name + ".csv", form_title, res)