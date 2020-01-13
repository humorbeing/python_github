import subprocess as sp


sp.call('ls')
sp.call(['ls', '-l'])
# sp.call(['exit', '0'])
with open('test.txt', 'w') as op:
    sp.call(['ls', '-l'], stdout=op)

output = sp.check_output(['echo', 'hello world'])
print(output)

baby = sp.Popen('gnome-system-monitor')
print(baby.poll())
# baby.wait()
baby.kill()
print(baby.poll())
sp.check_call('gnome-system-monitor')