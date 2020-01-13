import sys

if len(sys.argv) < 2:
    print("missing a command-line argument")
    exit(1)

print("hello, {}".format(sys.argv[1]))
exit(0)

#exit 0 is successs, exit else is failure
# cmd: echo $? can get what exit value is.
