import subprocess
print(' - 1'*5)

output = subprocess.call('ls', shell=True)
print(output)
print(' - 2'*5)
output = subprocess.check_output('ls', shell=True)
print(output)
print(' - 3'*5)
output = subprocess.check_call('ls', shell=True)
print(output)



import subprocess

for _ in range(64):
    subprocess.call('python main.py')