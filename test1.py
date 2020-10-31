

n = 8
s = [1,1]
if n < 0:
	print("nothing")
elif n ==0:
	print("1")
elif n == 1:
	print("2")
else:
	for i in range(2, n):
		s.append(s[i-1]+s[i-2]) 
		print(s)
	print(sum(s))