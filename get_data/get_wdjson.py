import json

with open("./wendadui.json",'r') as file:
    file_content = file.read()
    data = json.loads(file_content)
    tmp_data =[]
    all_data = []
    for item in data['conversations']:
        if item['role'] =='user':
            tmp_data = []
            tmp_data.append(item)
        else:
            tmp_data.append(item)
            all_data.append(tmp_data)

    print(all_data)
