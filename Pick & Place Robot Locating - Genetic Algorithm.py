"""
Modules used:
deap - Provide tools for GA optimisation
numpy - Provide tools for data science
math - Provide basic maths utilities
matplotlib.pyplot - provide utilities for plotting
"""
from deap import creator, base, tools, algorithms
import random
import numpy as np
import math
import matplotlib.pyplot as plt

"""
Class to describe the Robot in the workcell. 
Weight is approximated from the robots load and arm weight. 
Max_radius is the "reach" of the robot.
Min_radius is the minimum "reach" of the robot.
Base_Dimensions is the length of the base of the robot (known to be square).
"""
class Robot:
    def __init__(self, max_radius, min_radius, base_dimensions, weight, x, y, z):
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.base_dimensions = base_dimensions
        self.weight = weight
        self.x = x
        self.y = y
        self.z = z

"""
Class to describe the Conveyor belt in the workcell. 
X, Y and Z provide coordinates for its corners.
"""
class ConveyorBelt:
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

"""
Class to describe the Conveyor belt in the workcell. 
X, Y and Z provide coordinates for its corners.
"""
class Fixture:
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2

"""
Function to calculate the distance travelled by the robot
"""
def Distance(robot, fixture):
    # Calculate the 3D distance from the robot to each corner of the fixture
    distances = [math.sqrt((x - robot.x)**2 + (y - robot.y)**2 + (z - robot.z)**2) for x in [fixture.x1, fixture.x2] for y in [fixture.y1, fixture.y2] for z in [fixture.z1, fixture.z2]]
    return max(distances)

"""
Function to calculate the max energy for the robot to intreract with the fixture
"""
def FixtureEnergyUsage(robot, fixture):
    distance = Distance(robot, fixture)
    g = 9.81
    efficiency = 0.8
    energy = g*distance*robot.weight*efficiency
    return energy

"""
Function to calculate the max energy for the robot to intreract with the conveyor
"""
def ConveyorEnergyUsage(robot, conveyor):
    # Calculate the 3D distance from the robot to each point along the edge of the conveyor within the robot's reach
    distances = []
    for x in np.linspace(conveyor.x1, conveyor.x2, num=100):
        for y in [conveyor.y1, conveyor.y2]:
            for z in [conveyor.z1, conveyor.z2]:
                distance = math.sqrt((x - robot.x)**2 + (y - robot.y)**2 + (z - robot.z)**2)
                if robot.min_radius <= distance <= robot.max_radius:
                    distances.append(distance)
    ## Robot out of range
    if not distances:
        return float('inf')
    max_distance = max(distances)

    # Calculate the energy usage based on the maximum distance
    g = 9.81
    efficiency = 0.8
    energy = g * max_distance * robot.weight * efficiency
    return energy

"""
Function to check that all fixture related constraints are adhered to
"""
def CheckOverlapFixture(robot, fixture, clearance=150):
    # Calculate the distance from the robot to each corner of the fixture
    distances = [math.sqrt((x - robot.x)**2 + (y - robot.y)**2) for x in [fixture.x1, fixture.x2] for y in [fixture.y1, fixture.y2]]

    # Check if the robot's base is inside the fixture
    if any(abs(x - robot.x) <= robot.base_dimensions/2 + clearance and abs(y - robot.y) <= robot.base_dimensions/2 + clearance for x in [fixture.x1, fixture.x2] for y in [fixture.y1, fixture.y2]):
        return -1 

    # Check if the robot's arm can reach all corners of the fixture and they are not closer than the minimum radius
    if all(robot.min_radius <= distance <= robot.max_radius for distance in distances):
        return 0  
    
    return -1

"""
Function to check that all conveyor related constraints are adhered to
"""
def CheckOverlapConveyor(robot, pickup_zone, place_zone, fixture, clearance=150):
    # Check if the robot's base is inside the pickup or place zone
    for conveyor in [pickup_zone, place_zone]:
        for x in np.linspace(conveyor.x1, conveyor.x2, num=100):  
            if abs(x - robot.x) <= robot.base_dimensions/2 + clearance:
                return -1 
    # Check if any point along the width of the pickup or place zone intersects with the fixture
    for conveyor in [pickup_zone, place_zone]:
        for x in np.linspace(conveyor.x1, conveyor.x2, num=100):  
            for y in [conveyor.y1, conveyor.y2]:
                if fixture.x1 < x < fixture.x2 and fixture.y1 < y < fixture.y2:
                    return -1  
    # Check if the pickup and place zones are within the robot's arm reach
    for conveyor in [pickup_zone, place_zone]:
        for x in np.linspace(conveyor.x1, conveyor.x2, num=100):  
            distance = math.sqrt((x - robot.x)**2 + (conveyor.y1 - robot.y)**2)
            if not (robot.min_radius < distance <= robot.max_radius):
                return -1  
    # Check if the pickup and place zones overlap each other
    if abs(pickup_zone.y1 - place_zone.y1) < clearance:
        return -1  

    return 0

"""
Function to calculate the closest fixture to the robot (initial debugging step for visual)
"""
def ClosestFixture(robot):
    min_distance = float('inf')
    closest_fixture = None
    for fixture in fixtures:
        distance = math.sqrt((robot.x - (fixture.x2 + fixture.x1)/2)**2 + (robot.y - (fixture.y2 + fixture.y1)/2)**2 + (robot.z - (fixture.z2 + fixture.z1)/2)**2)
        if distance < min_distance:
            min_distance = distance
            closest_fixture = fixture
    return closest_fixture

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 2430)

robot = Robot(1103,301,343,135,1200, 1700, 1977)
fixtures = [Fixture(765, 1215, 1680, 1810, 0, 612),
            Fixture(765, 1215, 1450, 1560, 0, 612),
            Fixture(765, 1215, 1200, 1330, 0, 612)
            ]
pickup_zone = ConveyorBelt(650, 1150, 450, 750, 0, 600)
place_zone = ConveyorBelt(650, 1150, 450, 750, 0, 600)
hof_num = 20

"""
Function to create individuals so that each generated coordinate being within their own range
"""
def CreateIndividual():
    return [random.uniform(0, 2430),  # Robot's x-coordinate
            random.uniform(0, 6006),  # Robot's y-coordinate
            robot.z,  # Robot's z-coordinate
            random.uniform(0,2030),  # Conveyor's x1-coordinate
            random.uniform(450,5706),  # Conveyor's y1-coordinate
            ]

toolbox.register("individual", tools.initIterate, creator.Individual, CreateIndividual)
toolbox.register("attr_float", random.uniform, 0, 2430)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

"""
Function to calculate the energy and penalties for the current configuration
"""
def EvalEnergy(individual):
    energy = 0
    penalty = 0
    robot.x = individual[0]
    robot.y = individual[1]
    robot.z = individual[2]
    pickup_zone.x1 = individual[3]
    pickup_zone.y1 = individual[4]
    pickup_zone.x2 = pickup_zone.x1+400
    pickup_zone.y2 = pickup_zone.y1+300

    place_zone.x1 = individual[3]
    place_zone.y1 = individual[4]-150
    place_zone.x2 = place_zone.x1+400
    place_zone.y2 = place_zone.y1-300

    # Calculate energy for each robot-fixture pair
    for fixture in fixtures:
        energy += FixtureEnergyUsage(robot, fixture)

    # Calculate energy for the robot-pickup zone interaction
    energy += ConveyorEnergyUsage(robot, pickup_zone)

    # Calculate energy for the robot-place zone interaction
    energy += ConveyorEnergyUsage(robot, place_zone)

    # Check if the robot overlaps with all fixtures
    if not all(CheckOverlapFixture(robot, fixture) == 0 for fixture in fixtures):
        penalty += 2500000
    
    # Check if the robot overlaps with the pickup zone
    if not all(CheckOverlapConveyor(robot, pickup_zone,place_zone, fixture) == 0 for fixture in fixtures):
        penalty += 2500000

    return energy + penalty,

toolbox.register("evaluate", EvalEnergy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=0, up=2430, indpb=0.45)
#toolbox.register("select", tools.selBest)
#toolbox.register("select", tools.selRoulette)
#toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("select", tools.selStochasticUniversalSampling)
#toolbox.register("select", tools.selRandom)

pop = toolbox.population(n=60)
hof = tools.HallOfFame(hof_num)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(population=pop,
                               toolbox=toolbox,
                               cxpb=0.5,
                               mutpb=0.45,
                               ngen=2500,
                               stats=stats,
                               halloffame=hof,
                               verbose=True)

"""
Function to plot Workcell
"""
def PlotWorkcell(robot, fixtures, pickup_zone, place_zone, position, placement):
    robot.x = position[0]
    robot.y = position[1]
    pickup_zone.x1 = position[3]
    pickup_zone.y1 = position[4]
    pickup_zone.x2 = pickup_zone.x1+400
    pickup_zone.y2 = pickup_zone.y1+300
    place_zone.x1 = position[3]
    place_zone.y1 = position[4]-150
    place_zone.x2 = place_zone.x1+400
    place_zone.y2 = place_zone.y1-300

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the robot's maximum reach
    max_circle = plt.Circle((robot.x, robot.y), robot.max_radius, color='g', alpha=0.5)  # Set radius to robot's max reach
    ax.add_artist(max_circle)

    # Plot the robot's minimum reach
    min_circle = plt.Circle((robot.x, robot.y), robot.min_radius, color='r', alpha=0.5)  # Set radius to robot's min reach
    ax.add_artist(min_circle)

    # Plot each fixture
    for fixture in fixtures:
        fixture_rect = plt.Rectangle((fixture.x1, fixture.y1), fixture.x2 - fixture.x1, fixture.y2 - fixture.y1, fill=None)
        ax.add_artist(fixture_rect)

    # Plot the conveyor belt
    conveyor_rect = plt.Rectangle((pickup_zone.x1, 0), pickup_zone.x2 - pickup_zone.x1, 6006, fill=True, color='purple')
    ax.add_artist(conveyor_rect)

    # Plot the pickup zone
    pickup_rect = plt.Rectangle((pickup_zone.x1, pickup_zone.y1), pickup_zone.x2 - pickup_zone.x1, pickup_zone.y2 - pickup_zone.y1, fill=True, color='pink')
    ax.add_artist(pickup_rect)

    # Plot the place zone
    place_rect = plt.Rectangle((place_zone.x1, place_zone.y1), place_zone.x2 - place_zone.x1, place_zone.y2 - place_zone.y1, fill=True, color='orange')
    ax.add_artist(place_rect)

    # Set the limits of the plot to the maximum x and y coordinates
    ax.set_xlim(0, 2430)
    ax.set_ylim(0, 6006)

    ax.set_aspect('equal')
    plt.title(f'Placement {placement+1}')
    plt.show()


for i in range(hof_num):
    print(f"Placement {i+1}:")
    print(f"Robot coordinates: (X: {hof[i][0]}, Y: {hof[i][1]}, Z: {hof[i][2]})")
    print(f"Conveyor coordinates: (X1: {hof[i][3]}, Y1: {hof[i][4]}, X2: {hof[i][3]+400}, Y2: {hof[i][4]+300} for pickup zone, X1: {hof[i][3]}, Y1: {hof[i][4]-150}, X2: {hof[i][3]+400}, Y2: {hof[i][4]-450} for place zone)")
    print(f"Energy usage: {toolbox.evaluate(hof[i])[0]}")
    PlotWorkcell(robot, fixtures, pickup_zone, place_zone, hof[i], i)
    print(f"Robot is closest to Fixture {fixtures.index(closestFixture(robot))+1}")
    print()

