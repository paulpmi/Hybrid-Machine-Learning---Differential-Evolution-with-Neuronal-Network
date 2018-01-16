import random
import math
import numpy
import xlrd
import pandas
from copy import deepcopy
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

    def readCSV(self, filename):
        #with open('filename.csv', 'rb') as f:

        return pandas.read_excel(filename)[:899]


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
        self.outputNodes = self.normiliseOutput(sheet.values.tolist())
        self.input = numpy.asarray(self.input, dtype=numpy.float64)

        # input nodes change to have 1s
        self.inputNodes = []
        for i in self.input:
            l = i.flatten()
            l = numpy.append(1, l)
            self.inputNodes.append(l)

        self.outputNode = numpy.asarray(self.outputNodes, dtype=numpy.float64)

        self.x = []
        for i in range(self.neuronsInput * self.neuronsHidden):
            self.x.append(random.uniform(-1, 1))

        for layer in range(self.nrHidden):
            for neuron in range(self.neuronsHidden * self.neuronsHidden):
                self.x.append(random.uniform(-1, 1))

        for layer in range(self.neuronsHidden * self.neuronsOutput):
            self.x.append(random.uniform(-1, 1))

        self.sizeInput = len(self.inputNodes)

        self.x = numpy.asarray(self.x, dtype=numpy.float64)

        self.y = self.outputNode
        self.yOutputNode = random.uniform(-1, 1) # this is the actual output node that after the activation gives a result
        self.size = len(self.x)

    def fitness(self):
        fit = 0

        candidate = 0
        for inputN in self.inputNodes:
            #print("LOG INPUT NODES")
            #print(inputN.shape)
            output = []
            for i in range(self.neuronsHidden):
                output.append(self.activate(inputN, self.x, i, i+3))
            #print("LOG INITIAL OUTPUT: ")
            #print(output)

            beforeSynapses = deepcopy(output)
            position = self.neuronsHidden*self.neuronsInput
            for i in range(self.nrHidden-1):
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
        return fit

    def checkSolution(self, proposition):

        proposition = [1] + proposition
        print("LOG Proposition")
        print(proposition)
        position = 0

        hiddenLayer1Activated = []
        for i in range(self.neuronsHidden):
            hiddenLayer1Activated.append(self.activate(proposition, self.x, position, position + len(proposition)))
            position += len(proposition)

        output = []
        beforeSynapses = deepcopy(hiddenLayer1Activated)
        for i in range(self.nrHidden - 1):
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
        return output

    def activate(self, x, synapses,  startPosition, endPosition):
        s = numpy.dot(x, synapses[startPosition: endPosition])
        return 1 / (1 + numpy.exp(-s))

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
                normalisedData.append(j/1000)
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
                trialVector.x[i] = donorVector[i]
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
        i = random.randint(0, self.size - 1)
        factor = 0.5
        donorVector = self.equation(parent1, parent2, parent3, factor)

        return donorVector


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

    def evaluate(self, p):
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
            childCandidate = candidate.crossover(parent1, parent2)
            child = candidate.mutate(childCandidate, parent1, parent2)

            # print("Mutated")
            locationP1 = self.findParent(parent1)
            locationP2 = self.findParent(parent2)
            if parent1.fitness() < parent2.fitness():
                self.population[locationP1] = child
            else:
                self.population[locationP2] = child

    def reunion(self, toAdd):
        self.sizePop = self.sizePop + toAdd.size
        self.population = self.population + [toAdd]

    def selection(self, n, p):
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
        for k in range(no):
            parent1, parent2, parent3 = random.sample(self.population.population, 3)
            donorVector = self.population.population[k].mutate(parent1, parent2, parent3)
            trialVector = self.population.population[k].crossover(parent1, donorVector)
        offspring.evaluate(self.p)
        # self.population.selectionDE(trialVector, self.p)
        self.population.reunion(trialVector)
        self.population.selection(self.noInd, self.p)

    def run(self):
        for k in range(self.generations):
            print(k)
            self.iteration()
        return self.population.best(5, self.p)

    def writeData(self, filename, data):
        f = open(filename, 'w')
        try:
            for i in data:
                f.write(str(i))
                f.write(' ')
        except:
            f.write(str(data))



a = Algorthm(30, 30, 10)

solution = a.run()
a.writeData("Learning.txt", solution[0].x)
print(solution[-1].x, solution[-1].fitness())



f = open("checking4.txt", 'w')
#for i in range(len(solution)):
output = []
for sol in solution:
    output.append(str(sol.checkSolution([139.62, 0.452])))
f.write(str(output)) # output is: 141.38
#f.write(' ')
