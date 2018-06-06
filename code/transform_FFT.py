import sys

time = float(sys.stdin.readline())
numTest = int(sys.stdin.readline())
print(time/numTest)
print(numTest)

limit = sys.stdin.readline() # Maximos
print(limit, end="")

maxMat = eval(sys.stdin.readline())
l = [ sum(map(lambda x: float(x[i]), maxMat))/numTest for i in range(numTest) ]

print(l)
##################

limit = sys.stdin.readline() # Minimos
print(limit, end="")

minMat = eval(sys.stdin.readline())
l = [ sum(map(lambda x: float(x[i]), minMat))/numTest for i in range(numTest) ]

print(l)
##################

limit = sys.stdin.readline() # Abcissas dos Maximos
print(limit, end="")

abcMat = eval(sys.stdin.readline())
l = [ sum(map(lambda x: float(x[i]), abcMat))/numTest for i in range(numTest) ]

print(l)
##################

limit = sys.stdin.readline() # Maximo de fun U

l = sys.stdin.readline()

##################

limit = sys.stdin.readline() # Medio de fun U
print(limit, end="")

l = sys.stdin.readline()

print(l, end="")
##################

limit = sys.stdin.readline() # Minimo de fun U

l = sys.stdin.readline()

