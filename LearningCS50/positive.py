def main():#another ':'
    i = get_positive_int()
    print("{} is a positive integer".format(i))

def get_positive_int(): #don't need void
    while True:
        print("n is ", end="")
        n = int(input())
        if n >= 1:
            break
    return n

if __name__ == "__main__": #it works without this line.
    main()
# 使用main的原因是， get_positive_int 方法 要放在 被call之前
# code会变得很乱。已英语来理解，main就是主程序，说明main里面的
# 重要。为了可读性，损失了 长度。
