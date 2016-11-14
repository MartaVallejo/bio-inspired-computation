import numpy as np
import random


Input = np.array(([4,1],[5, 2], [3, 4], [4,2], [1, 5]), dtype=float)
ExpectedResult = np.array(([17], [29], [25],[20], [26]), dtype=float)


Input = Input/np.amax(Input, axis=0)
ExpectedResult = ExpectedResult/100 #MaInput test score is 100


class Neural_Network(object):

    def __init__(self):

        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.weight1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.weight2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self, Input):

        self.z2 = np.dot(Input, self.weight1)
        self.a2 = self.activationFunction(self.z2)
        self.z3 = np.dot(self.a2, self.weight2)
        Result = self.activationFunction(self.z3)
        return Result

    def sigmoid(self, z):

        return 1/(1+np.exp(-z))

    def activationFunction(self, Input):
        return Input * (Input > 0)

    def costFunction(self, Input, ExpectedResult):

        self.Result = self.forward(Input)
        cost = 0.5*sum((ExpectedResult-self.Result)**2)
        return cost


    def genAlg(self):

        bestweight1 = self.weight1
        bestweight2 = self.weight2
        bestCost = self.costFunction(Input, ExpectedResult)
        bestR = self.Result
        testNum = 50000;
        mutateChance = 0.5
        improveRate = 0.1
        for num in range(0, testNum):
            output = self.Result
            totEResult = 0;
            totResult = 0;

            for m in range(0, 4):
                totEResult = totEResult + ExpectedResult[m]
            for m in range(0, 4):
                totResult = totResult + (self.Result[m])

            for m in range(0, 2):
                for n in range(0,2):
                    randNum = random.uniform(0, 1)
                    if randNum < mutateChance :
                        mutation = self.costFunction(Input, ExpectedResult) * improveRate
                        if totResult < totEResult:
                            self.weight1[m,n] = bestweight1[m,n] + mutation
                        else:
                            self.weight1[m,n] = bestweight1[m,n] - mutation
            for m in range(0, 2):
                randNum = random.uniform(0, 1)
                if randNum < mutateChance:
                    mutation = self.costFunction(Input, ExpectedResult) * improveRate
                    if totResult < totEResult:
                        self.weight2[m] = bestweight2[m] + mutation
                    else:
                        self.weight2[m] = bestweight2[m] - mutation
            tempCost = self.costFunction(Input, ExpectedResult)
            if tempCost < bestCost :
                bestCost = tempCost
                bestweight1 = self.weight1
                bestweight2 = self.weight2
                bestR = self.Result
            else :
                self.weight1 = bestweight1
                self.weight2 = bestweight2
            print(tempCost, bestCost)
            print("OUTPUT")
            print(output)
        print(bestR)

network =  Neural_Network()
network.genAlg()
