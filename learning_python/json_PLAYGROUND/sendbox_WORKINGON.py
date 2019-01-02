import json


#some JSON

x = '{"name":"John", "age":30, "city":"New York"}'

#parse x:

y = json.loads(x)

# the result is python dictionary

print(y['age'])


# a python object (dictionary):

x = {
    "name": "John",
    'age': 30,
    'city': 'New York'
}

# convert into json:
y = json.dumps(x)

# the result is a json strong:

print(y)

# import json

print(json.dumps({"name": "John", "age": 30}))
print(json.dumps(["apple", "bananas"]))
print(json.dumps(("apple", "bananas")))
print(json.dumps("hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))

x = {
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

print(json.dumps(x))
# x=50000.0000001
print(json.dumps(x, indent=4))
print(json.dumps(x, indent=4, separators=(". ", " = ")))