
from jMetalPy.jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jMetalPy.jmetal.algorithm.multiobjective.spea2 import SPEA2

from jMetalPy.jmetal.operator import SBXCrossover, PolynomialMutation
from jMetalPy.jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jMetalPy.jmetal.problem.multiobjective.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from jMetalPy.jmetal.util.termination_criterion import StoppingByEvaluations

from jMetalPy.jmetal.lab.visualization.plotting import Plot
from jMetalPy.jmetal.core.solution import FloatSolution
from jMetalPy.jmetal.util.solution import get_non_dominated_solutions
import matplotlib.pyplot as plt
import numpy as np

from jMetalPy.jmetal.core.quality_indicator import InvertedGenerationalDistance

problem = ZDT1(number_of_variables=50)

max_evaluations = 1e4

algorithm1 = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability= 1.0/50.0, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

algorithm2 = SPEA2(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability= 1.0/50.0, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

algorithm1.run()
algorithm2.run()
solutions1 = algorithm1.get_result()
solutions2 = algorithm2.get_result()

front1 = get_non_dominated_solutions(solutions1)
front2 = get_non_dominated_solutions(solutions2)

nsgaii_front = Plot.get_points(front1)
spea2_front = Plot.get_points(front2)

if len(nsgaii_front[0]) == 3:
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(*zip(*nsgaii_front), c='b', s=5, label='NSGA-II')
    ax.scatter3D(*zip(*spea2_front), c='r', s=5, label='SPEA2')

elif len(nsgaii_front[0]) == 2:
    plt.scatter(*zip(*nsgaii_front), c='b', s=3, label='NSGA-II')
    plt.scatter(*zip(*spea2_front), c='r', s=3, label='SPEA2')

plt.legend()
plt.show()