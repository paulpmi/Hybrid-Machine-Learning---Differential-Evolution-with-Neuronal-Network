import numpy
import math
import random
import pandas
from copy import deepcopy
import sympy
import mpmath

numpy.seterr(all='raise')


class Individual:

    def __init__(self):
        self.neuronsInput = 1
        self.neuronsOutput = 1
        self.neuronsHidden = 2
        self.nrHidden = 2

        self.fit = 0

        self.inputLayer = []
        self.hiddenLayers = []
        self.outputLayer = []

        self.inputLayer = 2 * numpy.random.random((self.neuronsInput, self.neuronsHidden)) - 1
        for i in range(self.nrHidden-1):
            weights = 2 * numpy.random.random((self.neuronsHidden, self.neuronsHidden)) - 1
            self.hiddenLayers.append(weights)

        self.outputLayer = 2 * numpy.random.random((self.neuronsHidden, self.neuronsOutput)) - 1

        self.inputLayer = numpy.asarray(self.inputLayer)
        self.outputLayer = numpy.asarray(self.outputLayer)

        self.virtualInput = [[i] for i in range(900)]
        self.virtualOutput = [math.sin(i) for i in range(900)]

        self.input = self.readCSV('../input.xlsx', end=900)  # end between 900-1400

        self.test = self.readCSV('../actual_test_values.xlsx', end=2077)

        data = pandas.ExcelFile('../output.xlsx')
        sheet = data.parse('Sheet1')
        self.outputNodes = self.normiliseOutput(sheet.values.tolist()[:900])  # 900-1400

        self.input = numpy.asarray(self.input, dtype=numpy.float64)

        self.inputNodes = []
        for i in self.input:
            l = i.flatten()
            l = numpy.append(1, l)
            self.inputNodes.append(l)

        self.testData = []
        self.test = numpy.asarray(self.test, dtype=numpy.float64)
        for i in self.test:
            l = i.flatten()
            l = numpy.append(1, l)
            self.testData.append(l)

        self.outputNode = numpy.asarray(self.outputNodes, dtype=numpy.float64)

    def readCSV(self, filename, start=0, end=1400):
        return pandas.read_excel(filename)[start:end]

    def normiliseOutput(self, data):
        normalisedData = []

        for d in data:
            for j in d:
                normalisedData.append(j/100)
        return normalisedData

    def activate(self, nodeVal, weights):
        return numpy.dot(nodeVal, weights)

    def tahn(self, s):
        #return (numpy.exp(s) - numpy.exp(-s)) / (numpy.exp(s) + numpy.exp(-s))
        #s = [math.ceil((a*1e15)/1e15) for a in s]
        #s = numpy.asarray(s)
        return 1 / (1 + numpy.exp(-s))

    def fitness(self):
        fit = 0
        candidate = 0
        for inputN in self.virtualInput:
        #for inputN in self.inputNodes:
            try:
                inputLayerOutput = self.tahn(self.activate(inputN, self.inputLayer))

                hiddenLayerOutput = []
                for i in range(self.neuronsHidden - 1):
                    hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                    inputLayerOutput = deepcopy(hiddenLayerOutput)

                output = self.tahn(self.activate(hiddenLayerOutput, self.outputLayer))
                fit += (self.outputNode[candidate] * 100 - output * 100) ** 2
            except FloatingPointError:
                print("ENTERED ERROR")
                fit += 1000
            candidate += 1

        self.fit = fit
        return self.fit

    def reMutate(self):
        index_value = random.sample(list(enumerate(self.inputLayer)), 2)
        for i in index_value:
            self.inputLayer[i[0]] = 2 * numpy.random.random() - 1

        index_value = random.sample(list(enumerate(self.outputLayer)), 2)
        for i in index_value:
            self.outputLayer[i[0]] = 2 * numpy.random.random() - 1

        for i in range(len(self.hiddenLayers)):
            index_value = random.sample(list(enumerate(self.hiddenLayers[i])), 2)
            for j in index_value:
                self.hiddenLayers[i][j[0]] = 2 * numpy.random.random() - 1

    def NNStyle(self):
        error = []
        candidate = 0
        print("STARTED")
        for inputN in self.inputNodes:
            try:
                inputLayerOutput = self.tahn(self.activate(inputN, self.inputLayer))

                hiddenLayerOutput = []
                for i in range(self.neuronsHidden-1):
                    hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                    inputLayerOutput = deepcopy(hiddenLayerOutput)

                output = self.tahn(self.activate(hiddenLayerOutput, self.outputLayer))
                error.append(self.outputNode[candidate] - output)
                candidate += 1

            except FloatingPointError:
                pass
        return error

    def checkSolution(self, inputN):
        try:
            inputLayerOutput = self.tahn(self.activate(inputN, self.inputLayer))

            print("STARTED")
            print(inputLayerOutput)

            hiddenLayerOutput = []
            for i in range(self.neuronsHidden - 1):
                hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                inputLayerOutput = deepcopy(hiddenLayerOutput)

            print(hiddenLayerOutput)

            output = self.tahn(self.activate(hiddenLayerOutput, self.outputLayer))
            print("ENDED")
            return output
        except FloatingPointError:
            return -10

    def testAlgoritm(self):

        f = open("test.txt", 'w')
        output = []
        print(self.inputNodes)
        print(self.testData)
        #for i in self.testData:
        for i in self.virtualOutput:
            output.append(str(self.checkSolution(i)))
        f.write(str(output))

        return output


class Population:

    def __init__(self, sizePopulation):
        self.sizePopulation = sizePopulation
        self.population = [Individual() for i in range(self.sizePopulation)]
        self.lastBest = 1
        self.currentBest = 1

    def evaluate(self):
        sum = 0
        for x in self.population:
            sum += x.fit
        return sum

    def reMutatePopulation(self):
        for i in range(len(self.population)):
            if 0.5 > numpy.random.random():
                self.population[i].reMutate()

    def equationInput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsInput):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.inputLayer[i][j] - candidate.inputLayer[i][j]) * Factor + parent1.inputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.inputLayer[i][j])
            list.append(l)
        return list

    def equationHidden(self, parent1, parent2, candidate, Factor):
        mutationProb = Factor/2
        mutatedLayers = []
        for i in range(parent1.neuronsHidden-1):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.hiddenLayers[i][j] - candidate.hiddenLayers[i][j]) * Factor + parent1.hiddenLayers[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.hiddenLayers[i][j])
            mutatedLayers.append(numpy.asarray(l))
        return mutatedLayers

    def equationOutput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsHidden):
            l = []
            for j in range(parent1.neuronsOutput):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.outputLayer[i][j] - candidate.outputLayer[i][j]) * Factor + parent1.outputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.outputLayer[i][j])
            list.append(l)
        return list

    def mutate(self, parent1, parent2, candidate):
        donorVector = Individual()
        factor = 2 * random.uniform(-1, 1) * self.lastBest/self.currentBest
        donorVector.inputLayer = numpy.asarray(self.equationInput(parent1, parent2, candidate, factor))
        donorVector.hiddenLayers = numpy.asarray(self.equationHidden(parent1, parent2, candidate, factor))
        donorVector.outputLayer = numpy.asarray(self.equationOutput(parent1, parent2, candidate, factor))

        return donorVector

    def crossover(self, individ1, donorVector):
        crossoverRate = 0.5

        trialVector = Individual()

        for i in range(len(individ1.inputLayer)):
            for j in range(len(individ1.inputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.inputLayer[i][j] = individ1.inputLayer[i][j]
                else:
                    trialVector.inputLayer[i][j] = donorVector.inputLayer[i][j]

        for i in range(len(individ1.hiddenLayers)):
            for j in range(len(individ1.hiddenLayers[i])):
                if random.random() > crossoverRate:
                    trialVector.hiddenLayers[i][j] = individ1.hiddenLayers[i][j]
                else:
                    trialVector.hiddenLayers[i][j] = donorVector.hiddenLayers[i][j]

        for i in range(len(individ1.outputLayer)):
            for j in range(len(individ1.outputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.outputLayer[i][j] = individ1.outputLayer[i][j]
                else:
                    trialVector.outputLayer[i][j] = donorVector.outputLayer[i][j]

        return trialVector

    def evolve(self):
        childred = []
        indexes = []
        for i in range(self.sizePopulation):
            #candidate = self.population[i]
            parents = random.sample(list(enumerate(self.population)), 3)
            parent1Index, parent1 = parents[0]
            parent2Index, parent2 = parents[1]
            candidateIndex, candidate = parents[2]

            while parent1 == candidate or parent2 == candidate or parent1 == parent2:
                parents = random.sample(self.population, 2)
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

            child = self.mutate(parent1, parent2, candidate)
            childCandidate = self.crossover(candidate, child)
            childred.append(childCandidate)

            indexes.append(parent1Index)
            indexes.append(parent2Index)
            indexes.append(candidateIndex)

        return childred, indexes

    def selection(self, children, candidatesIndexes):
        for i in children:
            self.population[candidatesIndexes[0]].fitness()
            self.population[candidatesIndexes[1]].fitness()
            self.population[candidatesIndexes[2]].fitness()
            i.fitness()
            if self.population[candidatesIndexes[0]].fit > i.fit:
                self.population[candidatesIndexes[0]] = i
            else:
                if self.population[candidatesIndexes[1]].fit > i.fit:
                    self.population[candidatesIndexes[1]] = i
                else:
                    if self.population[candidatesIndexes[2]].fit > i.fit:
                        self.population[candidatesIndexes[2]] = i

    def best(self, n):
        aux = sorted(self.population, key=lambda Individual: Individual.fitness())
        return aux[:n]


class Algorithm:

    def __init__(self, sizePop, generations):
        self.population = Population(sizePop)
        self.sizePop = sizePop
        self.generations = generations

    def iteration(self):

        donorVector, indexes = self.population.evolve()
        self.population.selection(donorVector, indexes)
        offspringError = self.population.evaluate()
        self.population.lastBest = self.population.currentBest
        print(self.population.best(1)[0].fit)
        self.population.currentBest = self.population.best(1)[0].fit[0]
        print("LOG Global Error")
        print(offspringError/self.sizePop)

        #if self.population.currentBest >= offspringError/self.sizePop - 1:
        #    self.population.reMutatePopulation()

    def testRun(self):
        file = open("Final_Appended.txt", "a")
        file.write('\n')

        for k in range(self.generations):
            print(k)

            self.iteration()
        return self.population.best(10)


a = Algorithm(30, 100)


a.testRun()
#print(a.population.best(1)[0].NNStyle())


print("DEBUG")
print(a.population.best(1)[0].inputLayer)
print(a.population.best(1)[0].hiddenLayers)
print(a.population.best(1)[0].outputLayer)

print(a.population.best(1)[0].testAlgoritm())
print("vs")
print(a.population.best(1)[0].testData)
