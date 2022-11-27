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

For **K** vehicles where $K=\{1,2,...,|k|\}$:

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
    annotations.append("Central Location")
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


def generate_problem(title="cap_vehicle_routing_roblem"):
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


"""
def add_objective_function(
    problem, x, locations, vehicles=NUM_VEHICLES, num_locations=NUM_LOCATIONS
):
    problem += pulp.lpSum(
        locations[i][j] * x[i][j][k] if i != j else 0
        for k in range(vehicles)
        for j in range(num_locations)
        for i in range(num_locations)
    )
    return problem


def add_constraints(problem, x, vehicles=NUM_VEHICLES, num_locations=NUM_LOCATIONS):

    print(np.array(x).shape)

    # apply constraint (2)
    for jdx in range(num_locations):
        problem += pulp.lpSum(
            x[idx][jdx][kdx] if idx != jdx else 0
            for idx in range(num_locations)
            for kdx in range(num_locations)
        )

    # apply constraint (3)
    first = 0
    for kdx in range(vehicles):
        problem += (
            pulp.lpSum(x[first][jdx][kdx] for jdx in range(1, num_locations)) == 1
        )
        problem += (
            pulp.lpSum(x[idx][first][kdx] for idx in range(1, num_locations)) == 1
        )

    # apply constrait (4)
    for k in range(vehicles):
        for j in range(num_locations):
            problem += (
                pulp.lpSum(x[i][j][k] if i != j else 0 for i in range(num_locations))
                - pulp.lpSum(x[j][i][k] for i in range(num_locations))
                == 0
            )

    return problem
"""
```




    '\ndef add_objective_function(\n    problem, x, locations, vehicles=NUM_VEHICLES, num_locations=NUM_LOCATIONS\n):\n    problem += pulp.lpSum(\n        locations[i][j] * x[i][j][k] if i != j else 0\n        for k in range(vehicles)\n        for j in range(num_locations)\n        for i in range(num_locations)\n    )\n    return problem\n\n\ndef add_constraints(problem, x, vehicles=NUM_VEHICLES, num_locations=NUM_LOCATIONS):\n\n    print(np.array(x).shape)\n\n    # apply constraint (2)\n    for jdx in range(num_locations):\n        problem += pulp.lpSum(\n            x[idx][jdx][kdx] if idx != jdx else 0\n            for idx in range(num_locations)\n            for kdx in range(num_locations)\n        )\n\n    # apply constraint (3)\n    first = 0\n    for kdx in range(vehicles):\n        problem += (\n            pulp.lpSum(x[first][jdx][kdx] for jdx in range(1, num_locations)) == 1\n        )\n        problem += (\n            pulp.lpSum(x[idx][first][kdx] for idx in range(1, num_locations)) == 1\n        )\n\n    # apply constrait (4)\n    for k in range(vehicles):\n        for j in range(num_locations):\n            problem += (\n                pulp.lpSum(x[i][j][k] if i != j else 0 for i in range(num_locations))\n                - pulp.lpSum(x[j][i][k] for i in range(num_locations))\n                == 0\n            )\n\n    return problem\n'



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
        num_vehicles=1,
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
        self.problem = generate_problem()
        x, x_1d = generate_x(len(locations))
        self.x = x
        self.x_1d = x_1d

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
```

## Minimization Problem
