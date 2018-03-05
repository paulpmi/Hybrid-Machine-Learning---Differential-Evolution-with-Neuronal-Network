import csv

f = open("../Final_Appended_first.txt").readlines()


""" Test first output """

newGeneration = 0
xmlValues = []
finalValues = []
finalValues.append(["Value" + str(i) for i in range(17)])
realValues = []
for i in range(0, len(f)):
    values = []
    if len(f[i].split(',')) > 1:
        for g in f[i].split(','):
            number = ""
            for q in g:
                if q != '.':
                    try:
                        int(q)
                        number += q
                    except ValueError:
                        pass
                else:
                    number += '.'
            if number is not "":
                values.append(float(number))

        if newGeneration == 17:
            newGeneration = 0
            #for item in xmlValues:
            #wr = csv.writer(resultFile, dialect='excel')
            #wr.writerow([item, ])
            finalValues.append(xmlValues)
            print(xmlValues) # write these to excel
            xmlValues = []
        xmlValues.append(values[0])
        #print("Here: ", values)
        newGeneration += 1
    elif len(f[i].split(',')) == 1:
        if f[i].split(',')[0] != "\n":
            number = ""
            for q in f[i]:
                if q != '.':
                    try:
                        int(q)
                        number += q
                    except ValueError:
                        pass
                else:
                    number += '.'
            if number is not "" and float(number) not in realValues:
                realValues.append(float(number))
                #realValues.append(f[i])
                #print("Real: ", f[i])

#print(xmlValues) # write these to excel

print(finalValues)
finalValues.append(realValues)
lenValues = len(finalValues)

header = ["Output" + str(i) for i in range(lenValues)]
header = ["Start"] + header

with open('output2.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.DictWriter(myfile, fieldnames=header)
    writer.writeheader()
    wr = csv.writer(myfile)
    wr.writerows(finalValues)


finalValues = zip(*finalValues)

with open('output.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    writer = csv.DictWriter(myfile, fieldnames=header)
    writer.writeheader()
    wr = csv.writer(myfile)
    wr.writerows(finalValues)
    #wr.writerows(realValues)

print("Real:", realValues)


"""
for value in finalValues:
    for item in value:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(item)
"""
