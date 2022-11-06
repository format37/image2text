# This script sends image file to flask server

import requests
from datetime import datetime as dt

path = 'sample.jpg'
url = 'http://localhost:20000/request'
files = {'file': open(path, 'rb')}
start_time = dt.now()
print(start_time)
r = requests.post(url, files=files)
end_time = dt.now()
print(r.text)
print(end_time)
print(end_time - start_time)
