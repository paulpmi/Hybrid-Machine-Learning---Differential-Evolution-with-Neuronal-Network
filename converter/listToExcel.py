import csv
import math

f = open("../disection/test.txt").readlines()

header = ["Output" + str(i) for i in range(len(f))]
header = ["Start"] + header

print(f)
for i in f:

    print(len(i.split(',')))

    data = []
    for j in i.split(','):
        number = ""
        for k in j:
            if k == '-':
                number += k
            if k != '.':
                try:
                    float(k)
                    number += k
                except ValueError:
                    pass
            else:
                number += '.'
        try:
            data.append(float(number))
        except ValueError:
            pass

    print(data)
    alldata = []

    alldata.append(data)
    finalValues = zip(*alldata)
    with open('outputTest.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        writer = csv.DictWriter(myfile, fieldnames=header)
        writer.writeheader()
        wr = csv.writer(myfile)
        wr.writerows(finalValues)

    virtualOutput = [math.sin(i) for i in range(100)]
    alldata.append(virtualOutput)
    finalValues = zip(*alldata)
    with open('virtualOutputTest.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        writer = csv.DictWriter(myfile, fieldnames=header)
        writer.writeheader()
        wr = csv.writer(myfile)
        wr.writerows(finalValues)

    break
