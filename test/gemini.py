import requests

url = "https://docs.newapi.pro/v1beta/models"

response = requests.request("GET", url, headers = {
  "Authorization": "Bearer sk-r6objgSP2CfAXs5ImJE3FN3q1vP0hrYil07UmgJrInUqyvQG"
})

print(response.text)