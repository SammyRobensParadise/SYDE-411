# SYDE 411 Optimization Project

## Team members:

- Sammy Robens-Paradise, 20709541
- Wassim Maj , 21048694
- Tim He, 20779429
- <full_name> , <student_number>


## Import Dependencies

Dependencies are `numpy` `matplotlib` and `scipy`. You can install both dependencies (assuming you have `Python >= 3` ) using pip:

- `pip install numpy`
- `pip install matplotlib`
- `pip install scipy`
- `pip install pulp`



```python
import numpy as np
import matplotlib.pyplot as plt
import pulp

np.random.seed(18945)
# Some formating options
%config InlineBackend.figure_formats = ['svg']
```

## Problem Definition

The problem will be formulated as follows: Given 1 bus with a maximum capacity, $m$, we want to minimize the distance that the bus has to travel to $n$ locations (nodes) to pick up a total number of $P$ Our objective function is to minimize the total distance travelled, which is in turn a function of the distance to each location and the whether the locaction is connected to another location. This is commonly referred to as the **Vehicle Routing Problem**



## Problem to Minimize

For **K** vehicles where $K=\{1,2,...,k\}, k \gt 0$:

$G=(V,E)$ is a graph of the location and routes of the vehicles

$V=\{0,1,...n\}$ is a collection of nodes (locations) where $n_0$ is the start and end location for $K$ vehicles

$E$ is the set of edges $e_{ij} = (i,j)$ connecting each node

$c_{ij}$ is the cost (distance) between $i$ and $j$

### Variables:

$x_{ij}=
\begin{cases}
1: \text{the path goes from city i to j}\\
0: \text{otherwise}
\end{cases}$

$u_i - u_j + C*x_{ij} \leq C - d_j, C = N/K$
$\begin{cases}
u_i: \text{order that site i is visited}\\
d_j: \text{the cost to visit city j}, 0 \leq u_i \leq C - d_j, \forall i \in V \setminus \{0\}
\end{cases}$

### Objective Function:

(1) - $min\sum_{i=0}^{n}\sum_{j\neq{i}, j=0}^{n}{c_{ij}}{x_{ij}}$
$\begin{cases}
c_{ij}: \text{distance from city i to city j}\\
x_{ij}: \text{whether there's a path between i and j}
\end{cases}$

### With the following constraints:

(2) - $\sum_{i\in{V}}{x_{ij} = 1}, \forall j \in V \setminus \{0\}$

(3) - $\sum_{j\in{V}}{x_{ij} = 1}, \forall i \in V \setminus \{0\}$

(4) - $\sum_{i\in{V}}{x_{i0} = K}$

(5) - $\sum_{j\in{V}}{x_{0j} = K}$

### Where
- (1) is the _objective function_ 
- (2,3) constrains the (1)such that a location that is not the start and end location can only be visited by one vehicle
- (4) constrains (1) such that the start location is the first and that every vehicle starts there.
- (5) constrains (1) problem such the last location is the same as the first location and every vechicle must end there.


## Define Default Problem Constants


```python
# CONSTANTS

# delare constants to seed the data model
# the number of locations EXCLUDING the central starting and ending location
NUM_LOCATIONS = 9
NUM_VEHICLES = 2
GRID_SIZE = {"x": 1000, "y": 1000}
SEED = 18945
CENTRAL_LOCATION="Central Location"


# possible distances are
# - "manhattan"
# - "euclidean"
# - "chebyshev"
DISTANCE_METHOD = "manhattan"
```

## Create Help Methods


```python
def distance(
    p1,
    p2,
    method="manhattan",
):
    """Calculate the distance between two 2D points

    Parameters:
        p1 (ndarray) length == 2
        p2 (ndarray) length == 2
        method (string) "manhattan" | "euclidean" | "chebyshev"

    Returns:
        d (int) 2D distance between p1 and p2
    """
    # make sure that the distance is between only a coordinate
    assert len(p1) == len(p2) == 2
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    d = 0
    # if the locations are the same then we can automatically say the distance is 0
    if x1 == x2 and y1 == y2:
        return d
    if method == "manhattan":
        d = np.absolute(x1 - x2) + np.absolute(y1 - y2)
    elif method == "euclidean":
        d = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
    elif method == "chebyshev":
        d = np.max([np.absolute(x2 - x1), np.absolute(y2 - y1)])
    return d


def generate_data(
    num_locations=NUM_LOCATIONS,
    grid_size=GRID_SIZE,
    seed=None,
    distance_method="manhattan",
):
    """Generate the data for a given problem

    Args:
        num_locations (_int_, optional) Defaults to NUM_LOCATIONS.
        grid_size (_dict_, optional). Defaults to GRID_SIZE.
        seed (_int_, optional). Defaults to None.
        distance_method (str, optional). Defaults to "manhattan".

    Returns:
        locations: array of locations
        distances: n by n matrix of distances to each location from each location
        annotations: array of strings annotated coordinates of each location
    """
    # seed our data so that we have reproducible data
    if seed != None:
        np.random.seed(seed)
    locations = []
    annotations = []
    central_location_coord = [0, 0]
    # the central location is taken to be (0,0)
    locations.append(central_location_coord)
    annotations.append(CENTRAL_LOCATION)
    for _ in range(num_locations):
        x_coord = np.random.randint(-1 * grid_size["x"] / 2, grid_size["x"] / 2)
        y_coord = np.random.randint(-1 * grid_size["y"] / 2, grid_size["y"] / 2)
        node_coord = [x_coord, y_coord]
        label = "(" + str(x_coord) + "," + str(y_coord) + ")"
        locations.append(node_coord)
        annotations.append(label)
    locations = np.array(locations)
    distances = []
    for c1 in locations:
        local_distance = []
        for c2 in locations:
            d = distance(c1, c2, method=distance_method)
            local_distance.append(d)
        distances.append(local_distance)

    return locations, np.array(distances), annotations


def _validate_distances_(distances):
    assert (
        distances.shape[0] == distances.shape[1] == NUM_LOCATIONS + 1
    ), "Expected Distance matrix to be square and to equal NUM_LOCATIONS + 1"
    for i, _ in enumerate(distances):
        assert distances[i][i] == 0, "Distance diagonals cannot be non-zero"


def generate_lp_problem(title="cap_vehicle_routing_roblem"):
    """Generates a PuLP problem

    Args:
        title (str, optional): _description_. Defaults to "cap_vehicle_routing_roblem".

    Returns:
        PuLP Problem: Minimization Problem `LpProblem`
    """
    return pulp.LpProblem(title, pulp.LpMinimize)


def not_none(v):
    """returns true of the value is not `None`

    Args:
        v (`Any`): value

    Returns:
        Boolean: `True` or `False`
    """
    if v != None:
        return True
    return False


def generate_x(location_count):
    """generates n*n x variables

    Args:
        location_count (`int`): location_count

    Returns:
        n x n list `x`: variables in their n x n form
        1 x (n x x) list `x_1d`: variables in their list form
    """
    x = [
        [
            pulp.LpVariable("x%s_%s" % (i, j), 0, 1, pulp.LpBinary) if i != j else None
            for j in range(location_count)
        ]
        for i in range(location_count)
    ]
    x_1d = []
    for row in x:
        for val in row:
            x_1d.append(val)
    x_1d = list(filter(not_none, x_1d))
    return x, x_1d


def generate_u(location_points):
    """generate dummy `u` variables

    Args:
        location_points (ndarray): array of strings

    Returns:
        u: array of `LpVariables`
    """
    u = [
        pulp.LpVariable("u_%s" % (name), 0, len(location_points) - 1, pulp.LpInteger)
        for name in location_points
    ]
    return u


def objective_function(problem, x, distances):
    """applys the objective function to the problem

    Args:
        problem (LpProblem): a linear programming problem
        x (ndarray): x_ij
        distances (ndarrray): distances

    Returns:
        LpProblem: `LpProblem`
    """
    distances_1d = []
    for row in distances:
        for d in row:
            if d != 0:
                distances_1d.append(d)
    sum = []
    for idx, d in enumerate(distances_1d):
        ls = x[idx] * d
        sum.append(ls)
    cost = pulp.lpSum(sum)
    problem += cost
    return problem


def constraints(problem, x, location_points, num_vehicles):
    """Apply contraints to a problem

    Args:
        problem (LpProblem): `LpProblem`
        x (ndarray): x variables
        location_points (ndarray): array of locations
        num_vehicles (int): number of vehicles

    Returns:
        problem: `LpProblem`
    """
    x_transp = np.transpose(x)
    for idx, location in enumerate(location_points):
        max_visits = 1
        if location == CENTRAL_LOCATION:
            max_visits = num_vehicles
        sum = list(filter(not_none, x[idx]))
        sum_transp = list(filter(not_none, x_transp[idx]))
        # constrain inbound connections
        problem += pulp.lpSum(sum) == max_visits
        # constrain outbound connections by taking the transpose
        # i.e x_0_1 --> x_1_0
        problem += pulp.lpSum(sum_transp) == max_visits
    return problem


def subtours(problem, x, u, location_points, num_vehicles):
    """Constrains subtours of a given VRP problem

    Args:
        problem (LpProblem): `LpProblem`
        x (ndarray): x variables
        u (ndarray): u dummy variables to handle inequalities
        location_points (ndarray): array of locations
        num_vehicles (int): number of vehicles
        
    Returns:
        problem: `LpProblem`
    """
    n = len(location_points) / num_vehicles
    for i, l1 in enumerate(location_points):
        for j, l2 in enumerate(location_points):
            if l1 != l2 and (l1 != CENTRAL_LOCATION and l2 != CENTRAL_LOCATION):
                problem += u[i] - u[j] <= (n) * (1 - (x[i][j])) - 1
    return problem
```

## Class Definition
Create a class `Problem` so that we can create multiple problems with various test iteratively and dynmaically
It can then be used as
```py

locations, distances, annotations = generate_data(
    num_locations=NUM_LOCATIONS,
    grid_size=GRID_SIZE,
    seed=SEED,
    distance_method=DISTANCE_METHOD,
)

problem = Problem(locations=locations, distances=distances, annotations=annotations)
```


```python
# PROBLEM CLASS DEFINITION
class Problem:
    def __init__(
        self,
        num_locations=NUM_LOCATIONS,
        num_vehicles=NUM_VEHICLES,
        grid_size=GRID_SIZE,
        seed=SEED,
        id=1,
        locations=None,
        distances=None,
        annotations=None,
    ):
        self.id = id
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        self.seed = seed
        _validate_distances_(distances)
        assert (
            len(locations) == self.num_locations + 1
        ), "Error: Incorrect number of locations created"
        self.distances = distances
        self.locations = locations
        self.annotations = annotations
        # create a Linear Programming Optimization problem
        self.problem = generate_lp_problem()
        # generate binary variable x
        x, x_1d = generate_x(len(locations))
        self.x = x
        # linearize x into a 1D array to make the math easier
        self.x_1d = x_1d
        # generate dummy variable u to eliminate subtours
        u = generate_u(self.annotations)
        self.u = u
        # add objective function to our problem
        self.problem = objective_function(self.problem, x_1d, distances)
        # apply constrains to our objective function
        self.problem = constraints(
            self.problem, self.x, self.annotations, self.num_vehicles
        )
        self.problem = subtours(self.problem, x, u, self.annotations, self.num_vehicles)

    def minimize(self, method="default"):
        self.problem.startClock()
        if method == "simplex":
            self.problem.solve(pulp.apis.GLPK(options=["--simplex"]))
        elif method == "default":
            self.problem.solve()
        self.problem.stopClock()
        return pulp.LpStatus[self.problem.status]

    def plot_locations(self):
        plt.figure()
        plt.suptitle("Scatter plot of locations for problem: " + str(self.id))
        plt.grid()
        colors = np.random.rand(len(self.locations))
        plt.scatter(self.locations[:, 0], self.locations[:, 1], c=colors)
        plt.xlabel("$x$ m")
        plt.ylabel("$y$ m")
        plt.xlim([-1 * self.grid_size["x"] / 2, self.grid_size["x"] / 2])
        plt.ylim([-1 * self.grid_size["y"] / 2, self.grid_size["y"] / 2])
        for i, label in enumerate(self.annotations):
            plt.annotate(label, (self.locations[:, 0][i], self.locations[:, 1][i]))
        plt.show()

    def print_problems_state(self):
        print("Minimization problems for problem of id: " + str(self.id))
        for i, problem in enumerate(self.problems):
            print("Problem:" + str(i) + " =======")
            print(problem)
```

## Create Input Data



```python
locations, distances, annotations = generate_data(
    num_locations=NUM_LOCATIONS,
    grid_size=GRID_SIZE,
    seed=SEED,
    distance_method=DISTANCE_METHOD,
)
```

## Create Problem and Solve


```python
P1 = Problem(locations=locations, distances=distances, annotations=annotations)
P1.plot_locations()
status = P1.minimize()
print(status)
```


    
![svg](project_files/project_14_0.svg)
    


    Welcome to the CBC MILP Solver 
    Version: 2.10.3 
    Build Date: Dec 15 2019 
    
    command line - /usr/local/lib/python3.9/site-packages/pulp/solverdir/cbc/osx/64/cbc /var/folders/kw/sd26wxpx6w97t0sv3_w7tggc0000gn/T/930442f4f32247d194cca8931facd012-pulp.mps timeMode elapsed branch printingOptions all solution /var/folders/kw/sd26wxpx6w97t0sv3_w7tggc0000gn/T/930442f4f32247d194cca8931facd012-pulp.sol (default strategy 1)
    At line 2 NAME          MODEL
    At line 3 ROWS
    At line 97 COLUMNS
    At line 782 RHS
    At line 875 BOUNDS
    At line 975 ENDATA
    Problem MODEL has 92 rows, 99 columns and 396 elements
    Coin0008I MODEL read with 0 errors
    Option for timeMode changed from cpu to elapsed
    Continuous objective value is 3786 - 0.00 seconds
    Cgl0004I processed model has 92 rows, 99 columns (99 integer (90 of which binary)) and 396 elements
    Cutoff increment increased from 1e-05 to 0.9999
    Cbc0038I Initial state - 16 integers unsatisfied sum - 4.2
    Cbc0038I Pass   1: suminf.    4.20000 (12) obj. 4124 iterations 17
    Cbc0038I Pass   2: suminf.    2.00000 (9) obj. 4734.8 iterations 22
    Cbc0038I Pass   3: suminf.    1.60000 (4) obj. 4480.8 iterations 15
    Cbc0038I Pass   4: suminf.    1.20000 (6) obj. 4689.2 iterations 16
    Cbc0038I Pass   5: suminf.    2.20000 (6) obj. 4849.6 iterations 15
    Cbc0038I Pass   6: suminf.    1.60000 (4) obj. 4685.2 iterations 18
    Cbc0038I Pass   7: suminf.    4.00000 (10) obj. 5795.2 iterations 32
    Cbc0038I Pass   8: suminf.    1.60000 (4) obj. 5377.6 iterations 31
    Cbc0038I Pass   9: suminf.    1.06667 (8) obj. 5587.87 iterations 28
    Cbc0038I Pass  10: suminf.    1.06667 (8) obj. 5404.93 iterations 14
    Cbc0038I Pass  11: suminf.    1.40000 (6) obj. 5635.2 iterations 12
    Cbc0038I Pass  12: suminf.    1.60000 (4) obj. 5741.2 iterations 26
    Cbc0038I Pass  13: suminf.    1.60000 (4) obj. 6495.6 iterations 24
    Cbc0038I Pass  14: suminf.    1.60000 (8) obj. 6268.8 iterations 22
    Cbc0038I Pass  15: suminf.    0.00000 (0) obj. 6040 iterations 17
    Cbc0038I Solution found of 6040
    Cbc0038I Cleaned solution of 6040
    Cbc0038I Before mini branch and bound, 43 integers at bound fixed and 0 continuous
    Cbc0038I Full problem 92 rows 99 columns, reduced to 46 rows 49 columns
    Cbc0038I Mini branch and bound improved solution from 6040 to 4804 (0.05 seconds)
    Cbc0038I Round again with cutoff of 4701.3
    Cbc0038I Pass  16: suminf.    4.20000 (12) obj. 4124 iterations 0
    Cbc0038I Pass  17: suminf.    1.91138 (10) obj. 4701.3 iterations 29
    Cbc0038I Pass  18: suminf.    1.60000 (4) obj. 4480.8 iterations 14
    Cbc0038I Pass  19: suminf.    1.20000 (6) obj. 4689.2 iterations 22
    Cbc0038I Pass  20: suminf.    3.43063 (12) obj. 4701.3 iterations 17
    Cbc0038I Pass  21: suminf.    2.40000 (6) obj. 4235.6 iterations 22
    Cbc0038I Pass  22: suminf.    4.40000 (13) obj. 4474.4 iterations 28
    Cbc0038I Pass  23: suminf.    3.48000 (11) obj. 4390.24 iterations 7
    Cbc0038I Pass  24: suminf.    2.20000 (7) obj. 4300.4 iterations 4
    Cbc0038I Pass  25: suminf.    2.40000 (6) obj. 4235.6 iterations 18
    Cbc0038I Pass  26: suminf.    2.40000 (6) obj. 4235.6 iterations 10
    Cbc0038I Pass  27: suminf.    4.40000 (13) obj. 4474.4 iterations 31
    Cbc0038I Pass  28: suminf.    3.48000 (11) obj. 4390.24 iterations 5
    Cbc0038I Pass  29: suminf.    2.20000 (7) obj. 4300.4 iterations 4
    Cbc0038I Pass  30: suminf.    2.40000 (6) obj. 4235.6 iterations 18
    Cbc0038I Pass  31: suminf.    2.40000 (6) obj. 4235.6 iterations 11
    Cbc0038I Pass  32: suminf.    4.40000 (13) obj. 4474.4 iterations 24
    Cbc0038I Pass  33: suminf.    3.48000 (11) obj. 4390.24 iterations 7
    Cbc0038I Pass  34: suminf.    2.20000 (7) obj. 4300.4 iterations 3
    Cbc0038I Pass  35: suminf.    2.40000 (6) obj. 4235.6 iterations 22
    Cbc0038I Pass  36: suminf.    2.40000 (6) obj. 4235.6 iterations 11
    Cbc0038I Pass  37: suminf.    4.40000 (13) obj. 4474.4 iterations 25
    Cbc0038I Pass  38: suminf.    3.48000 (11) obj. 4390.24 iterations 5
    Cbc0038I Pass  39: suminf.    2.20000 (7) obj. 4300.4 iterations 1
    Cbc0038I Pass  40: suminf.    2.40000 (6) obj. 4235.6 iterations 16
    Cbc0038I Pass  41: suminf.    2.40000 (6) obj. 4235.6 iterations 9
    Cbc0038I Pass  42: suminf.    4.40000 (13) obj. 4474.4 iterations 21
    Cbc0038I Pass  43: suminf.    3.48000 (11) obj. 4390.24 iterations 7
    Cbc0038I Pass  44: suminf.    2.20000 (7) obj. 4300.4 iterations 3
    Cbc0038I Pass  45: suminf.    2.40000 (6) obj. 4235.6 iterations 15
    Cbc0038I No solution found this major pass
    Cbc0038I Before mini branch and bound, 57 integers at bound fixed and 0 continuous of which 1 were internal integer and 0 internal continuous
    Cbc0038I Full problem 92 rows 99 columns, reduced to 35 rows 33 columns
    Cbc0038I Mini branch and bound did not improve solution (0.09 seconds)
    Cbc0038I After 0.09 seconds - Feasibility pump exiting with objective of 4804 - took 0.07 seconds
    Cbc0012I Integer solution of 4804 found by feasibility pump after 0 iterations and 0 nodes (0.09 seconds)
    Cbc0038I Full problem 92 rows 99 columns, reduced to 7 rows 6 columns
    Cbc0031I 14 added rows had average density of 39.714286
    Cbc0013I At root node, 14 cuts changed objective from 3786 to 4208 in 100 passes
    Cbc0014I Cut generator 0 (Probing) - 103 row cuts average 2.4 elements, 0 column cuts (0 active)  in 0.034 seconds - new frequency is 1
    Cbc0014I Cut generator 1 (Gomory) - 949 row cuts average 80.1 elements, 0 column cuts (0 active)  in 0.023 seconds - new frequency is 1
    Cbc0014I Cut generator 2 (Knapsack) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.015 seconds - new frequency is -100
    Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.002 seconds - new frequency is -100
    Cbc0014I Cut generator 4 (MixedIntegerRounding2) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.009 seconds - new frequency is -100
    Cbc0014I Cut generator 5 (FlowCover) - 4 row cuts average 2.0 elements, 0 column cuts (0 active)  in 0.019 seconds - new frequency is -100
    Cbc0014I Cut generator 6 (TwoMirCuts) - 165 row cuts average 22.2 elements, 0 column cuts (0 active)  in 0.012 seconds - new frequency is 1
    Cbc0014I Cut generator 7 (ZeroHalf) - 16 row cuts average 7.9 elements, 0 column cuts (0 active)  in 0.654 seconds - new frequency is -100
    Cbc0010I After 0 nodes, 1 on tree, 4804 best solution, best possible 4208 (1.55 seconds)
    Cbc0012I Integer solution of 4710 found by DiveCoefficient after 2601 iterations and 12 nodes (1.60 seconds)
    Cbc0004I Integer solution of 4584 found after 2799 iterations and 19 nodes (1.62 seconds)
    Cbc0038I Full problem 92 rows 99 columns, reduced to 81 rows 24 columns
    Cbc0038I Full problem 106 rows 99 columns, reduced to 92 rows 89 columns - too large
    Cbc0001I Search completed - best objective 4584, took 5134 iterations and 84 nodes (1.80 seconds)
    Cbc0032I Strong branching done 1156 times (10407 iterations), fathomed 11 nodes and fixed 16 variables
    Cbc0035I Maximum depth 12, 679 variables fixed on reduced cost
    Cuts at root node changed objective from 3786 to 4208
    Probing was tried 311 times and created 711 cuts of which 0 were active after adding rounds of cuts (0.051 seconds)
    Gomory was tried 307 times and created 1381 cuts of which 0 were active after adding rounds of cuts (0.043 seconds)
    Knapsack was tried 100 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.015 seconds)
    Clique was tried 100 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.002 seconds)
    MixedIntegerRounding2 was tried 100 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.009 seconds)
    FlowCover was tried 100 times and created 4 cuts of which 0 were active after adding rounds of cuts (0.019 seconds)
    TwoMirCuts was tried 307 times and created 544 cuts of which 0 were active after adding rounds of cuts (0.035 seconds)
    ZeroHalf was tried 100 times and created 16 cuts of which 0 were active after adding rounds of cuts (0.654 seconds)
    ImplicationCuts was tried 192 times and created 16 cuts of which 0 were active after adding rounds of cuts (0.001 seconds)
    
    Result - Optimal solution found
    
    Objective value:                4584.00000000
    Enumerated nodes:               84
    Total iterations:               5134
    Time (CPU seconds):             1.19
    Time (Wallclock seconds):       1.81
    
    Option for printingOptions changed from normal to all
    Total time (CPU seconds):       1.20   (Wallclock seconds):       1.82
    
    Optimal

