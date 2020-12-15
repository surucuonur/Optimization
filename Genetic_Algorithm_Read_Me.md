ENGG*6140 Optimization Techniques

The Travelling Salesman Problem Solved Using

Genetic Algorithms

Project Report

Students:

Waleed Hilal (0913914)

Onur Surucu ( 0968305 )

15 th of August 2020


## Table of Contents

- 1. Introduction
- 2. Problem Formulation & Suggested Solution
- 3. Case Studies
- 4. Results & Discussion
- 5. Conclusion
- 6. References
- 7. Appendix
- Figure 1: Basic structure and flow of Genetic Algorithms Table of Figures
- Figure 2: Schematic of how the crossover operation works in Genetic Algorithms
- Figure 3: Overview of the mutation operation
- Figure 4: Graph of training process with different population sizes
- Figure 5: The code snippet of genetic algorithm implemented in Python.
- Figure 6: The code snippet of genetic algorithm implemented in Python.
- Figure 7: The code snippet of genetic algorithm implemented in Python.
- Figure 8: The code snippet of genetic algorithm implemented in Python.
- Figure 9: The code snippet of genetic algorithm implemented in Python.
- Table 1: Average estimates converged to and time elapsed for different population sizes Table of Tables
- Table 2: Average estimates converged to for each population size after 1 minute and 5 minutes


## 1. Introduction

The Traveling Salesman Problem (TSP) is one of the most extensively researched
mathematical problems that until present day, has no optimal solution [1]. The problem entails
determining a route from a starting point, touring a given number of areas or cities whilst only
visiting each location once. Ultimately, the goal is to return to the initial starting point of the
route whilst maintaining a minimal travel distance.

Presently, this problem is classified as NP-hard since the only known optimal solution
involves the enumeration of each possible tour, where each tour is a permutation of the number
of areas that must be visited [2]. This technique is known as the Brute-Force approach, which
presents a computational problem, as the number of possible tour permutations is given by
(푁− 1 )!
2 , where _N_^ is the number^ of areas to visit [3]. As _N_^ proceeds to increase and becomes large,
the problem becomes impossible to solve. To demonstrate this, consider current computational
capabilities of identifying and processing a single tour per nanosecond, or a billion tours per
second [3]. In order to evaluate all the possible enumerations, it would require approximately
490 million years to converge to a solution.

Thus, it is necessary to employ some sort of algorithm in order to converge to a solution
more efficiently and swiftly. As previously stated, this problem is NP-hard and therefore, a
solution in polynomial time does not exist. Hence, it may be prudent to sacrifice a degree of
optimality in order to realistically arrive at a solution for this problem [3].

After review of available literature on the topic, it is evident that heuristic algorithms have
displayed the most promising results for solving the TSP [4]. A notable algorithm known as
Simulated Annealing (SA) is based on emulating the metallurgical physical process of slowly
cooling a metal, and is used to arrive at the global minimum of a cost function [5]. SA algorithm
begins with a random solution, in which a nearby solution is then possible through every
iteration. The new solution is compared to the current solution and replaces it if it is superior [3].
If it is not a better solution, it can still be chosen to become the current solution and is then
associated with a probability that depends on a temperature parameter that decreases as the
program runs. This results in inferior solutions having a less significant opportunity to replace the
current solution [3].

In this research, we explore the use of another optimization technique known as Genetic
Algorithms (GA) in order to arrive at an approximation of an optimal solution. GAs are based on
and inspired by the process of natural selection, which progressively breeds a population of
solutions over a series of generations using the Darwinian principle [6]. Genetic programming
uses a probabilistic algorithm and Darwinian principles of natural selection such as recombination
or crossover, mutation, gene duplication, gene deletion and various other biological mechanisms


in order to produce new and improved populations over generations under guidance of a fitness
measure [7].

## 2. Problem Formulation & Suggested Solution

The TSP as previously mentioned, involves determining the shortest route between an
arbitrary number of locations. There are several conditions however, one being that each
location may only be visited once and another that the ending point must be at the initial starting
point, meaning that the route is a closed loop. The distances between each location have been
provided as an _NxN adjacency_ matrix, or the cost matrix, where N represents the total number
of places to visit. Each element of the matrix _Xij_ represent the distance from location _i_ to location
_j_ , and the matrix is symmetrical, which implies that _Xij_ is equal to _Xji_. The dataset given to us that
we are trying to solve contains 48 locations ( _N_ = 48) and represents each of the 48 mainland
states of the United States of America.

Brute Force is currently the only optimal way to solve these problems, but the
computational effort and time required by this method increases exponentially as the number of
cities increases [8]. The approach involves simply calculating the total distance of each possible
route from the given cost matrix and enumerating over all the possible permutations, choosing
the one with the lowest total distance. The number of possible permutations is given by:

```
(푁− 1 )!
2 =
```
### ( 48 − 1 )!

### 2 =^1.^29 ×^10

(^59)
It is clear and evident from this value, that a program coded using a Brute Force approach would
run for a very large amount of time, depending on the computational power of the system
running it. The time required by this algorithm renders it impractical, as this problem can be
translated into different various real-world applications that would have to be much faster in
calculating an optimal feasible solution.
Genetic Algorithms are another optimization method that involve a sense of
computational intelligence in the search for an approximation of an optimal solution within a
reasonable amount of time [9]. These algorithms are inspired by the process of natural selection,
and they are a subset of random-based evolutionary algorithms. They are useful in scenarios
where the search space is very large or too complex for analytic treatment, which is true in the
case of the TSP, and have been shown to outperform random or systematic searches [10]. Initial
guesses of solutions known as chromosomes are first produced, which must undergo certain
processes such as crossover and mutation to produce further chromosomes. A fitness function is
defined and used to determine the fitness or cost of each chromosome, from which the fittest
will be used to reproduce more chromosomes. In this case, the fitness function is simply the


summation of the distances between each sequential city in the chromosome. The result is the
breeding of better chromosomes through an iterative heuristic approach to arrive at the best
possible solution. A visual representation of the flow of operations within a Genetic Algorithm is
clear in figure 1 below.

```
Figure 1 : Basic structure and flow of Genetic Algorithms
```
By selecting Python as our programming language, it was possible to program and code a
Genetic Algorithm that would be able to carry out all the functions involved in the algorithm’s
processes. Python was chosen due to its simplicity in comparison to other high-level languages,
as well as the vast availability of useful libraries that would help carry out the operations
necessary. Pandas is a well-known library that’s used in Python to carry out data analysis and
manipulation tasks and provides a framework for data structures. The symmetrical cost matrix
of the 48 states provided will be stored in the form of a Pandas data frame. From this data frame,
random permutations of possible routes will be generated with the help of another library,
Numpy. Numpy has a wide variety of useful functions that enable array and matrix operations in
a simple fashion, as well as the ability to create random permutations of a data frame which is
what will be happening in our case.

First, the program reads and stores the dataset as a variable which then initializes the
genotype size according to the provided matrix which is 48 in this case. One of the
hyperparameters that were selected was the population size which was set to be 20. A class was


used for the population which contained all the instances of all the permutations of route, and
another class was used for all the operations to perform on the instances of routes. After the
instance for each route has been initialized, random permutations of the existing instance were
generated and put through the fitness function. Their corresponding fitness value would
determine their probabilities of being selected to undergo further alterations such as mutation
and crossover in order to produce new combinations. An example of crossover occurring can be
seen in figure 2.

```
Figure 2 : Schematic of how the crossover operation works in Genetic Algorithms [11]
```
Another operation that occurs after a crossover is a random mutation. Here, after a crossover is
successful and produces a new child, the child is subjected to a random mutation in any of its
genes within the chromosome. Introduction of a random mutation ensures that the population
remains diverse, avoiding the possibility of the algorithm converging prematurely to a local
optimum [12]. An illustration of this occurring to an arbitrary gene within a chromosome can be
see in figure 3.

```
Figure 3 : Overview of the mutation operation [12]
```

## 3. Case Studies

Nowadays, the formulation of big data has become an issue for most of the real world
contexts such as computer wiring, vehicle routing, clustering a data array and manufacturing of
PCBs and many various industry problems [13]. Nature’s approaches have inspired the
researchers to discover new optimizing techniques such as the ant colony optimizer, water wave
optimizer, simulated annealing, coordinating particle swarm optimizer and many more. Many of
these techniques have been implemented and used in many real-world applications and
industries.

The Travelling Salesman Problem (TSP) has become an early proving ground of finding the
near-optimal tour for most of the new heuristic algorithms due to its simplicity and applicability.
In 15 years, the developed heuristic algorithms could expand the record for the largest nontrivial
TSP optimal solution from 318 to 2392 cities [14][15]. In short words, the success of the
traditional approaches leave less opportunity for new algorithms such as simulated annealing,
ant colony optimization and similar known approaches. On the other hand, genetic algorithms
have shown that they can compete with other algorithms, if one is willing to optimize a large
dataset which results a large running time price 푂(푁^2 ) [16][17].

The ant colony optimization (ACO) algorithm was introduced by Marco Dorigo in the early
1990 s [18]. The algorithm is inspired from ant colony’s food finding strategy. While exploring the
environment for food, the real ants leave pheromones along the path directing each other to the
source. They tend to lay down a stronger pheromone concentration on their trail if they believe
the probability of the path leads to the food source. Thus, the others will know what road to
choose along their way. To implement ACO into the TSP, the initial cities chosen are considered
as a nest. Then, each route to the neighbor cities will be weighted according to the distances.
After multiple iterations, among all the collected possible routes, the optimal one is picked by
the fitness function which is the total distance of the trail in our case.

The particle swarm is a population based stochastic optimization technique proposed by
Dr. Eberhart and Dr. Kennedy in 1995 [19]. Essentially, the emergent motion of bird flocking was
the inspiration of the optimizer. The particles represent the possible solutions whilst the optimal
solution is the food. The particles move through a multi-dimensional search space where the
particles are guided by their own best experience and that of its neighbors.

Simulated annealing was formulated in 1970 inspired by annealing in metallurgy [20]. The
inspiration came from the technique involving heating and cooling instantly to increase its
ductility and decrease its hardness. This process affects the temperature and the thermodynamic
free energy. The algorithm starts with a randomly picked solution, then a random nearby solution
is created in each iteration [3]. If the new solution is fitter, then it will be replaced as the current
solution [3]. If the previous one is worse, the assigned temperature value decides whether to pick
that solution or not. The lesser temperature value is being assigned to new solutions as the
algorithm spreads out, so the less chance is being given to the worse solutions. The worst


solutions are more freely considered at the beginning to avoid converging to the local minima
instead of the global minima [3].

## 4. Results & Discussion

Within the Genetic Algorithm, there were certain things that were fixed and absolute,
such as the number of cities that were used an example to determine the shortest distance route.
However, there were other various factors that had to be considered and tuned in order get the
performance desired out of the Genetic Algorithm. These are called hyperparameters and include
things such as the population size and the number of generations (or iterations) to evaluate.
Hyperparameters are involved with the actual learning of the algorithm and influence the
efficiency of the algorithm’s convergence.

The way the population size and number of generations parameters were chosen was
through trial and error, as there was no other practical way of doing so. Initially, a population size
was chosen and the number of generations to iterate over was kept at a relatively low number
to have the algorithm execute in as little time as possible. If the algorithm was still converging at
a decent pace before the end of the program, then that indicated that more iterations were
needed and as such the program was amended accordingly. However, a couple of trends were
observed with the variation of these two factors. As the size of the population is increased, the
algorithm takes a longer time to complete and the result produced is less accurate. The time
taken by the algorithm cannot be fixed, however we can converge to a better result in this
scenario if we increase the number of iterations at a cost of the algorithm taking even longer to
finish executing.

## Figure 4: Graph of training process with different population sizes

```
40000
```
```
50000
```
```
60000
```
```
70000
```
```
80000
```
```
90000
```
```
100000
```
```
110000
```
```
120000
```
```
130000
```
```
140000
```
```
1
```
(^22745367990511311357158318092035226124872713293931653391361738434069429545214747497351995425565158776103632965556781)
**Distance (km)
Epoch**
Population size: 40 Population size: 20 Population size: 10


A graphical representation is provided in figure 4 of the GA running and searching for a
feasible solution with three different population sizes. Three different population sizes were
trialled, which were 10, 20 and 40. It was determined throughout the course of this experiment
that a population size of 20 was the most optimal value as it results in the shortest distance in
the same number of epochs as the other population sizes, which is shown in figure 4. This
population size prevented the algorithm from taking too long from executing which occurs when
the population is too large, even though it may result in a better estimate. In the case of small
population sizes, such as 10 in this experiment, there is a lack of diversity within the solutions
and thus the solution space is restricted. This results in the algorithm stalling at the early stages
of when it is running and ultimately a less accurate solution upon convergence. It is important to
note that in figure 4, the number of epochs is not related to the time elapsed for each population
to converge.

For each population size, the time elapsed until the algorithm converges to a solution was
also recorded. This time is independent of the number of epochs and can be a measure of the
computational complexity of the algorithm. The main factor affecting the time it takes for the
algorithm to converge is the population size, whilst increasing the population size does not
necessarily result in converging to a better estimate. The time elapsed for each algorithm and its
respective population size was recorded, and the results are visible in table 1. From this table we
can see that a population size of 20 achieved the shortest distance out of all three population
sizes and was able to converge in under a minute. Increasing the population size to 40 did not
result in converging to a more optimal solution despite taking a longer time for the algorithm to
run. Having a population size of 10 led to faster convergence, taking a little over half of how long
the algorithm with a population size of 20 ran but sacrificing a small degree of accuracy.

```
Table 1 : Average estimates converged to and time elapsed for different population sizes
```
Another method of quantifying the efficiency of these algorithms, was to restrict the time
they could run for. This is useful as it allows us to determine the degree of accuracy the algorithm
may be able to achieve within a given time. This also demonstrates the practicality of this
algorithm if it were to be deployed in a real-life scenario. For example, if this algorithm were to
be utilized to determine the route of a delivery truck making many stops along its route, it would

```
Average values of 10 runs with 7000 epochs
Population size Fitness (total distance) Time
```
(^10) 46879.78 29.
20 45314.55 50.
(^40) 47862.33 102.


be very costly to have to wait for long periods of time to determine a feasible route. Results for
the three different population sizes discussed earlier are shown in table 2. When only allowed to
run for a minute, we can see in table 2 that the population size of 10 resulted in the shortest
distance solution. Having a population size of 20 results in a slightly higher distance, and the
population size of 40 leads to a much higher value for the distance estimate, rendering it useless
in comparison to the other two population sizes. Upon increasing the time allowed for the
algorithm to run, a population size of 20 yields the shortest distance and thus the most accurate
measurement. These findings support our hypothesis that considerations must be made for the
tuning of hyperparameters. Increasing the population size may result in more accurate estimate
at the cost of requiring a greater amount of time to converge to that better solution.

## Table 2: Average estimates converged to for each population size after 1 minute and 5 minutes

In real world applications, the industry has demand for this type of application but with
better performance and decreased computational cost. In order to meet these requirements,
either the algorithm’s structure or the execution technique should be changed. Python shows
forty times slower performance compared to lower level languages which ultimately affects the
computational effort [21]. However, for the sake of this project, Python has been used due to its
elegant syntax and rich libraries. Another alternative way to achieve less computational time by
using Python is running the built-in libraries and using multiple threads to add concurrency into
the algorithm’s computation. On the other hand, various modifications in the algorithm can lead
superior results. According to researches, one of the most important aspect of the genetic
algorithm is the crossover function and usage of fine-tuned crossover functions yield better
results [22].

## 5. Conclusion

A method for solving the Travelling Salesman Problem (TSP) was presented in this paper
utilizing Genetic Algorithms (GA). The GA demonstrated itself as an algorithm that was capable
of converging to a solution given that the hyperparameters such as the population size were
properly tuned. Other similar techniques that have been used to solve the TSP include simulated

```
Average values of 10 runs during certain time
Population size Optimal Solution (1 min) Optimal Solution (5 mins)
10 44359.44 4019 8.
20 46798.7 7 39238.
40 52267.11 42509.
```

annealing, ant colony optimization and more. It was shown that the GA was able to converge to
a much better solution depending on certain factors such as time and the number of epochs
allowed. A population size of 20 provided the lowest distance solution to the proposed problem.
A correlation between population size and the accuracy of the solution provided was displayed.
As the population size increases, the solution output by the algorithm also increases. However,
the downside is that as the population increases so does the time required by the algorithm to
finally converge. From the dataset provided of the 48 states of the United States of America, the
population size of 20 yielded the best performance overall in terms of robustness and accuracy.
In the future, improvements could be made by using a more efficient programming language or
increasing the computational capacity and power of the device running the algorithm. Thus, it
was shown that Genetic Algorithms display great potential for solving TSP problems in real world
applications and industry.


## 6. References

```
[1] R. Matai, S. Prakash and M. L. Marari, "Travelling Salesman Problem: An Overview of Applications,
Formulations, and Solution Approaches," in Traveling Salesman Problem, Theory and Applications ,
Shanghai, InTech, 2010.
```
```
[2] A. Homaifar, S. Guan and G. E. Liepins, "Schema Analysis of the Traveling Salesman Problem Using
Genetic Algorithms," Complex Systems, vol. 6, no. 2, pp. 533-552, 1992.
```
```
[3] M. Alhanjouri, "Optimization Techniques for Solving Travelling Salesman Problem," International
Journal of Advanced Research in Computer Science and Software Engineering, vol. 7, no. 3, 2017.
```
```
[4] I. M. Malib Muneeb Abid, "Heuristic Approaches to Solve Traveling Salesman Problem,"
TELKOMNIKA Indonesian Journal of Electrical Engineering, vol. 15, no. 2, pp. 390- 39 6, 2015.
```
```
[5] D. Bertsimas and J. Tsitsiklis, "Simulated Annealing," Statistical Science, vol. 8, no. 1, pp. 10-15,
1993.
```
```
[6] A. Ghost and S. Tsutsui, Advances in Evolutionary Computing, Springer, 2003.
[7] R. Riolo and B. Worzel, Genetic Programming Theory and Practice, Springer, 2003.
```
```
[8] K. Srinivasan, S. Satyajit, B. K. Behera and P. K. Panigrahi, "Efficient quantum algorithm for solving
travelling salesman problem: An IBM quantum experience," Quantum Physics, 2018.
```
```
[9] V. Dwivedi, T. Chauhan, S. Saxena and P. Agrawal, "Travelling Salesman Problem using Genetic," in
National Conference on Development of Reliable Information Systems, Techniques and Related
Issues (DRISTI) , 2012.
[10] P. Godefroid and S. Khurshid, "Exploring very large state spaces using genetic algorithms,"
International Journal on Software Tools for Technology Transfer, vol. 6, pp. 117-127, 2004.
[11] A. Dutta, Artist, Single Point Crossover: Genetic Algorithms. [Art]. Geek for Geeks.
```
```
[12] A. Kumar, "Genetic Algorithms, Geeks for Geeks," [Online]. Available:
https://www.geeksforgeeks.org/genetic-algorithms/. [Accessed 20 July 2020].
```
```
[13] J. K. Lenstra and A. H. G. Rinnooy Kan, "Some Simple Applications of the Travelling Salesman
Problem," Operational Research Quarterly, pp. 717-733, 1975.
```
```
[14] M. Padberg and H. Crowder, Solving Large-Scale Symmetric Travelling Salesman Problems to
Optimality, Management Science, 1980.
```
```
[15] M. Padberg and G. Rinaldi, "Optimization of a 532-city symmetric traveling salesman problem by
branch and cut," in Opertations Research Letters , 1987, pp. 1-7.
```

[16] W. Hui, "Comparison of several intelligent algorithms for solving TSP problem in industrial
engineering," _Systems Engineering Procedia,_ pp. 226-235, 2012.

[17] D. McGeoch and J. L. A. , The Traveling Salesman Problem: A Case Study in Local Optimization,
2008.

[18] N.-H. Chen, "An Ant Colony Optimization and Bayesian Network Structure Application for the
Asymmetric Traveling Salesman Problem," in _Intelligent Information and Database Systems_ , Berlin,
Springer Berlin Heidelberg, 2012, pp. 74-78.

[19] J. Kennedey and R. Eberhart, "Particle swarm optimization," in _Proceedings of ICNN'95 -
International Conference on Neural Networks_ , 1995, pp. 1942-1948.

[20] M. Pincus, "Operations Research," _Letter to the Editor—A Monte Carlo Method for the
Approximate Solution of Certain Types of Constrained Optimization Problems,_ 1970.

[21] M. Fourment and M. R. Gillings, "A comparison of common programming languages used in
bioinformatics," _BMC Bioinformatics,_ pp. 1471-2105, 2008.

[22] A. Shrestha, A. Mahmood and N. Tang, Improving Genetic Algorithm with Fine-Tuned Crossover
and Scaled Architecture, Hindawi Publishing Corporation, 2016.


## 7. Appendix

## Figure 5: The code snippet of genetic algorithm implemented in Python.

## Figure 6: The code snippet of genetic algorithm implemented in Python.


## Figure 7: The code snippet of genetic algorithm implemented in Python.

## Figure 8: The code snippet of genetic algorithm implemented in Python.


## Figure 9: The code snippet of genetic algorithm implemented in Python.


