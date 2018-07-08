import re

file = input('Enter name of file:')
handle = open(file)

count = 1
for line in handle:
    parts = line.split()
    print(str(count) + ". **", parts[0], "**  |  ", end="")
    count = count + 1
    for x in range(len(parts)):
        if x > 0:
            print(parts[x]," ", end="")
    print(" \n", end="")

    
    #print(str(count) + ". " + line, end="")
    #print("_"*len(line)
    
