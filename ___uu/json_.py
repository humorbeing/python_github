import json


data = {
    "name": "John",
    "age": 30,
    "married": True,
    "divorced": False,
    "children": ("Ann","Billy"),
    "pets": None,
    "cars": [
        {"model": "BMW 230", "mpg": 27.5},
        {"model": "Ford Edge", "mpg": 24.1}
    ]
}

json_string = json.dumps(data, indent=4)
print(json_string)

data_from_json = json.loads(json_string)

print(data_from_json['age'])


with open('data_file.json', 'w') as write_file:
    json.dump(data, write_file)

with open("data_file.json", 'r') as read_file:
    data_read = json.load(read_file)

print(data_read['age'])