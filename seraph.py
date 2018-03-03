import requests


username="UR14CS100" 
password="PASSWORD123" 


sess = requests.Session()

#To get the seraph page. We need the magic number to authenticate. 
#Without the magic number it wont work
try:
	a= sess.get('http://go.microsoft.com')
except:
	print("Cannot connect to the network. Try again later.")
	exit(0)

content = str(a.content)
try:
	magic = content.index('magic')
	magicVal=content[magic:magic+40].split('"')[2] #located the magic number
except :
		print("Already authenticated")	
		exit(0)
		
#Send the post with magic.
auth=requests.post("https://seraph.karunya.edu:1003", data={"username":username,"4Tredir":"http%3A%2F%2Fgo.microsoft.com%2F","magic":magicVal,"password":password })
print("Seraph Authenticated")
exit(0)

