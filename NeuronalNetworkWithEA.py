import random
import math
import numpy
import xlrd
import pandas
from copy import deepcopy
from sklearn.utils import shuffle
# NOTE: this is on D:, Community works on C:

class Problem:
    def __init__(self):
        self.minim = -5
        self.max = 5
        self.rminim = 0
        self.rmax = 0
        # self.readFromFile()

    def Ackleys(self, x, y):
        a = -20 * math.pow(math.exp(1), -0.2 * math.sqrt(0.5 * (x * x + y * y))) - math.pow(math.exp(1), 0.5 * (
        math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.exp(1) + 20
        return a

    def readData(self, fileName):
        f = open(fileName, 'r')
        data = []
        out = []
        i = 0
        for line in f:
            data.append(line.strip().split(','))
            q = []
            for j in range(0, len(data[i]) - 1):
                q.append(float(data[i][j]))
                data[i][j] = float(data[i][j])
            out.append(q)
            i += 1
        return out

    def readCSV(self, filename, start=0, end=1380):
        #with open('filename.csv', 'rb') as f:

        return pandas.read_excel(filename)[start:end]


class Individual:

    def __init__(self):
        self.neuronsInput = 3
        self.neuronsOutput = 1
        self.neuronsHidden = 5
        self.nrHidden = 5

        # normileze Input Nodes
        self.input = Problem().readCSV('input.xlsx')

        data = pandas.ExcelFile('output.xlsx')
        sheet = data.parse('Sheet1')
        self.outputNodes = self.normiliseOutput(sheet.values.tolist()[:1380])
        #self.outputNodes = shuffle(self.outputNodes)
        self.input = numpy.asarray(self.input, dtype=numpy.float64)
        #self.input = shuffle(self.input)

        # input nodes change to have 1s
        self.inputNodes = []
        for i in self.input:
            l = i.flatten()
            l = numpy.append(1, l)
            self.inputNodes.append(l)

        #print(self.inputNodes[:1][0])
        self.outputNode = numpy.asarray(self.outputNodes, dtype=numpy.float64)

        self.x = []
        for i in range(self.neuronsInput * self.neuronsHidden):
            self.x.append(random.uniform(-1, 1))

        for layer in range(self.nrHidden-1):
            for neuron in range(self.neuronsHidden * self.neuronsHidden):
                self.x.append(random.uniform(-1, 1))

        for layer in range(self.neuronsHidden * self.neuronsOutput):
            self.x.append(random.uniform(-1, 1))

        self.sizeInput = len(self.inputNodes)

        self.x = numpy.asarray(self.x, dtype=numpy.float64)

        self.y = self.outputNode
        self.yOutputNode = random.uniform(-1, 1) # this is the actual output node that after the activation gives a result
        self.size = len(self.x)

    def updateInput(self, inputFile, outputFile, start, end):
        self.input = Problem().readCSV(inputFile, start, end)
        data = pandas.ExcelFile(outputFile)
        sheet = data.parse('Sheet1')
        self.outputNodes = self.normiliseOutput(sheet.values.tolist()[start:end])
        self.input = numpy.asarray(self.input, dtype=numpy.float64)
        self.sizeInput = len(self.inputNodes)

        # input nodes change to have 1s
        self.inputNodes = []
        for i in self.input:
            l = i.flatten()
            l = numpy.append(1, l)
            self.inputNodes.append(l)

        self.outputNode = numpy.asarray(self.outputNodes, dtype=numpy.float64)
        self.y = self.outputNode

        #print("final")
        #print(self.input)
        #print(self.y)

    def newActivation(self, nodes, weights, position, nrNeuronsNextLayer):
        output = []
        pos = 0
        #print("LOG LEN NODES: ", len(nodes))
        for i in range(nrNeuronsNextLayer):
            pos = deepcopy(position) - nrNeuronsNextLayer
            nodeOutput = []
            for j in range(len(nodes)):
                pos += nrNeuronsNextLayer
                #print("HERE ", j, " ", pos)
                nodeOutput.append(nodes[j] * weights[pos])
            #print(pos, " ", position)
            position += 1
            output.append(nodeOutput)

        sumOutput = []
        for i in output:
            sumOutput.append(float("{0:.6f}".format(1 / (1 + numpy.exp(-sum(i))))))
        return sumOutput, pos

    def newActivationRelu(self, nodes, weights, position, nrNeuronsNextLayer):
        output = []
        pos = 0
        #print("LOG LEN NODES: ", len(nodes))
        for i in range(nrNeuronsNextLayer):
            pos = deepcopy(position) - nrNeuronsNextLayer
            nodeOutput = []
            for j in range(len(nodes)):
                pos += nrNeuronsNextLayer
                #print("HERE ", j, " ", pos)
                nodeOutput.append(nodes[j] * weights[pos])
            #print(pos, " ", position)
            position += 1
            output.append(nodeOutput)

        sumOutput = []
        for i in output:
            s = sum(i)
            if s < 0:
                sumOutput.append(0.01)
            else:
                sumOutput.append(float("{0:.4f}".format(s)))
        return sumOutput, pos

    def fitness(self):
        fit = 0

        candidate = 0
        for inputN in self.inputNodes:
            #print(inputN)
            #print("LOG INPUT NODES")
            #print(inputN.shape)
            #output = []
            #for i in range(0, self.neuronsHidden):
            #output.append()
            #output.append(self.activate(inputN, self.x, i, i+self.neuronsInput))

            output, position = self.newActivationRelu(inputN, self.x, 0, self.neuronsHidden)

            beforeSynapses = deepcopy(output)

            #position = self.neuronsHidden*self.neuronsInput

            for i in range(self.nrHidden-1):
                position += 1
                output, position = self.newActivationRelu(beforeSynapses, self.x, position, self.neuronsHidden)
                beforeSynapses = deepcopy(output)

            position += 1
            output, position = self.newActivation(beforeSynapses, self.x, position, self.neuronsOutput)
            #print(self.y[candidate], " ", output)
            fit += (self.y[candidate] - output[0]) ** 2
            candidate += 1
            #self.activateSoftMax()
            """
            for i in range(self.nrHidden):
                output = []
                for j in range(self.neuronsHidden):
                    output.append(self.activate(beforeSynapses, self.x, position, position+5))
                    position += 5
                beforeSynapses = deepcopy(output)
                #print("LOG TRIAL OUTPUT")
                #print(output)
            output = self.activate(output, self.x, position, position+5)
            #print("LOG SECOND OUTPUT")
            #print(output)
            fit += (self.y[candidate] - output)**2
            candidate += 1
            """
        return fit

    def checkSolution(self, proposition):

        #proposition = [1] + proposition
        proposition = numpy.append(1, proposition)
        #print("LOG PROPOSITION: ", proposition)
        output, position = self.newActivationRelu(proposition, self.x, 0, self.neuronsHidden)
        beforeSynapses = deepcopy(output)

        #print("LOG BEFORE SYNAPSES: ", beforeSynapses)

        for i in range(self.nrHidden - 1):
            position += 1
            output, position = self.newActivationRelu(beforeSynapses, self.x, position, self.neuronsHidden)
            beforeSynapses = deepcopy(output)
            #print("LOG BEFORE SYNAPSES: ", beforeSynapses)

        #print("LOG BEFORE SYNAPSES: ", beforeSynapses)
        position += 1
        output, position= self.newActivation(beforeSynapses, self.x, position, self.neuronsOutput)
        #print("ENDED ", output)
        """
        hiddenLayer1Activated = []
        for i in range(0, self.neuronsHidden*self.neuronsInput, 3):
            hiddenLayer1Activated.append(self.activate(proposition, self.x, position, position + len(proposition)))
            position += len(proposition)

        output = []
        beforeSynapses = deepcopy(hiddenLayer1Activated)
        for i in range(self.nrHidden):
            output = []
            for j in range(self.neuronsHidden):
                output.append(self.activate(beforeSynapses, self.x, position, position + 5))
                position += 5
            beforeSynapses = deepcopy(output)

        output = self.activate(output, self.x, position, position + 5)
        print("LOG CheckSolution Position")
        print(position, " ", len(self.x))
        print("LOG FINAL OUTPUT: ")
        print(output)
        """
        #print(output[0]*100, " ", proposition)
        return output[0]*100

    def activate(self, x, synapses,  startPosition, endPosition):
        s = numpy.dot(x, synapses[startPosition: endPosition])
        return 1 / (1 + numpy.exp(-s))

    def activateSoftMax(self, x, synapses,  startPosition, endPosition):
        s = numpy.dot(x, synapses[startPosition: endPosition])
        return 2 / (1 + numpy.exp(-2*s)) - 1

    def activateRelu(self, x, synapses,  startPosition, endPosition):
        s = numpy.dot(x, synapses[startPosition: endPosition])
        return s * (s > 0)

    def normaliseData(self, trainData):
        for j in range(1, len(trainData[0])):
            summ = 0.0
            for i in range(len(trainData)):
                summ += trainData[i][j]
            mean = summ / len(trainData)
            squareSum = 0.0
            for i in range(len(trainData)):
                squareSum += (trainData[i][j] - mean) ** 2
            deviation = numpy.sqrt(squareSum / len(trainData))
            for i in range(len(trainData)):
                trainData[i][j] = (trainData[i][j] - mean) / deviation
        return trainData

    def normiliseOutput(self, data):
        normalisedData = []

        for d in data:
            for j in d:
                #print(j[0])
                if j < 100:
                    normalisedData.append(j/100)
                else:
                    normalisedData.append(j/100)
        return normalisedData

    def computeOutputs(self, trainData):
        outputs = []
        noOutputs = 0
        for t in trainData:
            outputs.append(t[-1])
            noOutputs += 1
        #print("outputs ", outputs)
        return outputs

    def crossover(self, individ1, donorVector):
        crossoverRate = 0.5

        i = 0
        trialVector = Individual()
        while i < len(individ1.x):
            if random.random() > crossoverRate:
                trialVector.x[i] = individ1.x[i]
            else:
                trialVector.x[i] = donorVector.x[i]
            i += 1
        return trialVector

    def equation(self, parent1, parent2, parent3, Factor):
        l = []
        for i in range(parent1.size):
            nr = (parent2.x[i] - parent3.x[i]) * Factor + parent1.x[i]
            l.append(nr)
        return l

    def mutate(self, parent1, parent2, parent3):
        mutationProb = 0.5
        i = random.randint(0, 1)
        if i > mutationProb:
            donorVector = Individual()
            factor = random.uniform(-1, 1) # used be 0.5
            donorVector.x = self.equation(parent1, parent2, parent3, factor)

            return donorVector
        return parent1


#i = Individual()
#print(i.fitness())

class Population:
    def __init__(self, sizePop, noInd):
        self.noInd = noInd
        self.sizePop = sizePop
        self.population = [Individual() for _ in range(self.sizePop)]

    def findParent(self, p):
        for i in range(self.sizePop):
            if p == self.population[i]:
                return i
        return None

    def updatePopulationInput(self, inputFile, outputFile, start, end):
        updatedPop = []
        for i in self.population:
            i.updateInput(inputFile, outputFile, start, end)
            updatedPop.append(i)
        self.population = updatedPop

    def evaluate(self):
        sum = 0
        for x in self.population:
            sum += x.fitness()
        return sum

    def evolve(self):
        mutationProb = 0.5
        for i in range(self.sizePop):
            candidate = self.population[i]
            parents = random.sample(self.population, 2)
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            while parent1 == candidate or parent2 == candidate:
                parents = random.sample(self.population, 2)
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

            childCandidate = candidate.crossover(parent1, parent2)
            child = candidate.mutate(childCandidate, parent1, parent2)

            # print("Mutated")
            locationP1 = self.findParent(parent1)
            locationP2 = self.findParent(parent2)
            if parent1.fitness() < child.fitness():
                self.population[locationP1] = child
            elif parent2.fitness() < child.fitness():
                self.population[locationP2] = child
            """
            if parent1.fitness() < parent2.fitness():
                self.population[locationP1] = child
            else:
                self.population[locationP2] = child
            """

    def reunion(self, toAdd):
        self.sizePop = self.sizePop + toAdd.size
        self.population = self.population + [toAdd]

    def selection(self, n):
        if n < self.sizePop:
            self.population = sorted(self.population, key=lambda Individual: Individual.fitness())
            self.population = self.population[:n]
            self.sizePop = n

    def selectionDE(self, trialVector, p):
        for i in range(len(self.population)):
            if trialVector.fitness(p) < self.population[i].fitness():
                self.population[i] = trialVector

    def best(self, n, p):
        aux = sorted(self.population, key=lambda Individual: Individual.fitness())
        return aux[:n]


class Algorthm:
    def __init__(self, noInd, sizePop, generations):
        #self.iterNr = 0
        self.p = Problem()
        self.population = Population(sizePop, noInd)
        self.noInd = noInd
        self.sizePop = sizePop
        self.generations = generations

    def iteration(self):
        indexes = range(self.noInd)
        no = self.noInd // 2
        offspring = Population(self.sizePop, no)
        k = 1
        shuffle(self.population.population)
        for k in range(no):
            parent1, parent2, parent3 = random.sample(self.population.population, 3)
            donorVector = self.population.population[k].mutate(parent1, parent2, parent3)
            trialVector = self.population.population[k].crossover(parent1, donorVector)
        offspringError = self.population.evaluate()
        print("LOG Global Error")
        print(offspringError/self.sizePop)
        # self.population.selectionDE(trialVector, self.p)
        self.population.reunion(trialVector)
        self.population.selection(self.noInd)

    def run(self):
        finalOutput = []
        file = open("Final_Appended.txt", "a")
        file.write('\n')

        start = 345
        timesItChanged = 1

        for k in range(self.generations):
            print(k)
            self.iteration()
            if k % 50 == 0:
                #print("ENTERED")
                #print(self.p.readCSV('input.xlsx', k, k+1))
                candidate = self.p.readCSV('test_input.xlsx',  0, 10)
                candidate = numpy.asarray(candidate, dtype=numpy.float64)
                #shuffle(candidate)
                candidate = candidate[:10]
                bestSolutions = self.population.best(10, self.p)
                    #print(candidate)
                for i in range(len(candidate)):
                    output = []
                    output2 = []
                    #print(candidate)
                    #print(candidate[i])
                    for sol in bestSolutions:
                        output.append(str(sol.checkSolution(candidate[i])))
                        if i == 0:
                            output2.append(str(sol.checkSolution([29.75, 0.136])))
                            output2.append(str(sol.checkSolution([70.73, 0.535])))
                            output2.append(str(sol.checkSolution([72.15, 0.565])))
                            file2 = open("Final_Appended_second.txt", "a")
                            file2.write('OUTPUT 2. Solutions [30.4, 72.235555555555, 72.23]. Results are: ')
                            file2.write('\n')
                            file2.write(str(output2))
                            file2.write('\n')
                            file2.close()

                    #print("MIDDLE")
                    try:
                        file = open("Final_Appended_first.txt", "a")
                        file.write('\n')
                        file.write(str(output))
                        candidateoutput = self.p.readCSV('test_output.xlsx', 0, 10)
                        candidateoutput = numpy.asarray(candidateoutput, dtype=numpy.float64)
                        file.write('\n')
                        file.write(str(candidateoutput[i]))
                        file.write('\n')
                        file.close()
                    except Exception:
                        print("ERROR")
                    #print("ENDED")
                #finalOutput.append(output)
                #k = k + i
                #self.population.updatePopulationInput('input.xlsx', 'output.xlsx', start*timesItChanged, start*(timesItChanged+1))
            #self.population.updatePopulationInput('input.xlsx', 'output.xlsx', k, k+100)
        return self.population.best(10, self.p)

    def writeData(self, filename, data):
        f = open(filename, 'w')
        try:
            for i in data:
                f.write(str(i))
                f.write(' ')
        except:
            f.write(str(data))


a = Algorthm(30, 35, 500)

solution = a.run()
a.writeData("Learning.txt", solution[0].x)
print(solution[-1].x, solution[-1].fitness())

f = open("checking.txt", 'w')
output = []
for sol in solution:
    output.append(str(sol.checkSolution([24.05, 0.13])))
f.write(str(output)) # output is: 26.41

f = open("checking1.txt", 'w')
output = []
for sol in solution:
    output.append(str(sol.checkSolution([29.75, 0.136])))
f.write(str(output)) # output is: 30.4

f = open("checking2.txt", 'w')
output = []
for sol in solution:
    output.append(str(sol.checkSolution([70.73, 0.535])))
f.write(str(output)) # output is: 72.235555555555

f = open("checking3.txt", 'w')
output = []
for sol in solution:
    output.append(str(sol.checkSolution([70.73, 0.535])))
f.write(str(output)) # output is: 72.23

f = open("checking4.txt", 'w')
output = []
for sol in solution:
    output.append(str(sol.checkSolution([72.15, 0.565])))
f.write(str(output)) # output is: 71.8


