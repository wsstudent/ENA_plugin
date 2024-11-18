import http.client
import json

conn = http.client.HTTPSConnection("api.chatanywhere.tech")
payload = json.dumps({
   "model": "gpt-3.5-turbo",
   "prompt": "Say this is a test",
   "max_tokens": 7,
   "temperature": 0,
   "top_p": 1,
   "n": 1,
   "stream": False,
   "logprobs": None,
   "stop": "\n"
})
headers = {
   'Authorization': 'Bearer sk-nb6TMYvsD6KjllF5RC3i3Xl9tokeOe7rTdNMGnSmpI8ieog1',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))