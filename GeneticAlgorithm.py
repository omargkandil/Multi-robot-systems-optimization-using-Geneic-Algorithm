import numpy as np
import matplotlib.pyplot as plt
import math
import time
start_time = time.time()

# Define the number problem parameters
num_robots = 3 # The number of robots
num_waypoints = 6 # The number of waypoints for each robot
num_obstacles = 20 # Number of obstacles
max_x = 60 # The maximum x-coordinate of the environment
max_y = 60 # The maximum y-coordinate of the environment
robots_energy = 4000 # Bettery capacity for the robots

# Define the start and goal points for each robot as np arrays
start_points = np.array([[5, 1], [4, 2], [3, 3]])
goal_points = np.array([[50, 14], [49, 53], [19, 47]])

# Randomize obstacles
obstacle_points = [np.random.randint(0, max_x, size=(1, 2)).tolist()[0] for _ in range(num_obstacles)]
obstacle_points = np.array(obstacle_points)

# Define the genetic algorithm parameters
mutation_rate = 0.2 # The probability of mutating a gene
crossover_rate = 0.8 # The probability of performing crossover
num_generations = 500 # The number of generations
pop_size = 10 # The size of population

# Define the initialization function to create a random population
def initialize():
    # Initialize an empty list to store the population
    population = []
    # Loop through the population size
    for i in range(pop_size):
        # Initialize an empty list to store the paths for each robot
        paths = []
        # Loop through the number of robots
        for j in range(num_robots):
            # Generate random waypoints for the robot
            waypoints = np.random.randint(0, max_x+1, size=(num_waypoints, 2))
            # Append the waypoints to the paths list
            paths.append(waypoints)
        paths = smoothness_function(paths)
        # Append the paths to the population list
        population.append(paths)
    # Return the population as a np array
    return np.array(population)

# Define the fitness function to minimize the total distance travelled by each robot
def fitness_function(paths):
    # Initialize the total distance to zero
    total_distance = 0
    # Loop through each robot and its path
    for i in range(num_robots):
        # Calculate the distance from the start point to the first waypoint
        distance = np.linalg.norm(start_points[i] - paths[i][0])
        # Loop through the remaining waypoints
        for j in range(1, len(paths[i])):
            # Calculate the distance between two consecutive waypoints
            distance += np.linalg.norm(paths[i][j-1] - paths[i][j])
        # Calculate the distance from the last waypoint to the goal point
        distance += np.linalg.norm(paths[i][-1] - goal_points[i])
        # Add the distance to the total distance
        total_distance += distance
    # Return the negative of the total distance as the fitness value
    return -total_distance

# Define the smoothness function to accept waypoints depending on the angle between the vector from
# the current position to the goal and the vector from the current position to the next position
def smoothness_function(random_paths):
    # Initialize the smoothness score to zero
    paths = [[] for _ in range(num_robots)]
    # Loop through each robot and its path
    for i in range(num_robots):
        # Initialize a flag to indicate if the path is feasible
        paths[i].append(random_paths[i][0])
        # Loop through the waypoints from the second to the last one
        j=0
        while j in range(num_waypoints-1):
            # Get the current, next, and goal waypoints
            current = random_paths[i][j]
            next = random_paths[i][j+1]
            goal = goal_points[i]
            # Calculate the vector from the current to the next waypoint
            v1 = np.array([next[0] - current[0], next[1] - current[1]])
            # Calculate the vector from the current to the goal waypoint
            v2 = np.array([goal[0] - current[0], goal[1] - current[1]])
            # Calculate the angle between the two vectors in degrees
            dot_product = np.dot(v1, v2)
            norm_product = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
            # Calculate the distance of the current and the new path
            current_dist = calculate_distance(current,goal)
            next_dist = calculate_distance(next,goal)
            # Check if norm_product is zero before performing division
            if norm_product == 0:
                angle = 0
            else:
                # Ensure the input to arccos is within the valid range [-1, 1]
                angle = (np.degrees(np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))))
            # If the angle is greater than 30 degrees, the path is not feasible
            if angle < 40 and next_dist < current_dist: # angle for the smoothness of the pass
                collision = False
                for k in range(num_obstacles):
                    if (point_on_line_segment(current, next, obstacle_points[k])):
                        collision = True
                if (collision == False):
                    paths[i].append(next)
                    j += 1
                else:
                    random_paths[i][j + 1] = np.random.randint(0, max_x, 2)
            # If the path reched the goal then stop creating new paths
            elif current_dist == 0 and (next == goal).all():
                paths[i].append(next)
                j+=1
            # Path is not feasible then create a new path
            else:
                random_paths[i][j+1] = np.random.randint(0, max_x, 2)
    # Return the feasible paths
    return paths

def point_on_line_segment(point1, point2, check_point):
    x1, y1 = point1
    x2, y2 = point2
    x, y = check_point

    # Check if the point lies on the line formed by the other two points
    on_line = (min(x1, x2) <= x <= max(x1, x2)) and (min(y1, y2) <= y <= max(y1, y2))

    # Check if the cross product is zero
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    on_segment = abs(cross_product) < 1e-10
    # print(on_line and on_segment)
    return on_line and on_segment

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Define the selection function to select the best individuals for reproduction
def selection(population):
    # Initialize an empty list to store the fitness values
    fitness_values = []
    # Loop through the population
    for individual in population: # Loop over the array directly
        # Calculate the fitness value of the individual
        fitness = fitness_function(individual)
        # Append the fitness value to the list
        fitness_values.append(fitness)
    # Convert the list to a np array
    fitness_values = np.array(fitness_values)
    # Sort the fitness values in descending order
    sorted_indices = np.argsort(-fitness_values)
    # Select the top half of the population as the parents
    parents = population[sorted_indices[:pop_size//2]]
    # Return the parents as a np array
    return parents

# Define the crossover function to combine the genes of two parents
def crossover(parent1, parent2):
    # Initialize an empty list to store the paths for the child
    paths = []
    # Loop through the number of robots
    for i in range(num_robots):
        # Generate a random number for crossover
        r = np.random.rand()
        # If the number is less than the crossover rate, perform crossover
        if r < crossover_rate:
            # Choose a random crossover point
            point = np.random.randint(1, num_waypoints)
            # Combine the genes of the parents before and after the point
            waypoints = np.concatenate((parent1[i][:point], parent2[i][point:]), axis=0)
        # Otherwise, choose one of the parents as the child
        else:
            # Choose a random parent
            waypoints = parent1[i] if np.random.rand() < 0.5 else parent2[i]
        # Append the waypoints to the paths list
        paths.append(waypoints)
    # Return the paths as a np array
    return np.array(paths)

# Define the mutation function to introduce some variations in the genes
def mutation(child):
    # Loop through the number of robots
    for i in range(num_robots):
        # Loop through the number of waypoints
        for j in range(num_waypoints):
            # Generate a random number for mutation
            r = np.random.rand()
            # If the number is less than the mutation rate, perform mutation
            if r < mutation_rate:
                # Choose a random gene to mutate
                gene = np.random.randint(0, 2)
                # Generate a random value for the gene
                value = np.random.randint(0, max_x) if gene == 0 else np.random.randint(0, max_y)
                # Replace the gene with the value
                child[i][j] = value
    # Return the mutated child
    return child


# Define the main function to run the genetic algorithm
def main():
    # Initialize the population
    population = initialize()

    # Initialize an empty list to store the best fitness values at each generation
    best_fitness_values = []

    # Loop through the number of generations
    for i in range(num_generations):
        # Select the parents
        parents = selection(population)
        # Initialize an empty list to store the children
        children = []
        # Loop through the half of the population size
        for j in range(pop_size // 2):
            # Choose two random parents
            parent1 = parents[np.random.randint(0, pop_size // 2)]
            parent2 = parents[np.random.randint(0, pop_size // 2)]
            # Perform crossover to generate a child
            child = crossover(parent1, parent2)
            # Perform mutation to modify the child
            child = mutation(child)
            # Append the child to the children list
            children.append(child)
        # Replace the population with the parents and children
        population = np.concatenate((parents, children), axis=0)

        # Calculate the best fitness value in the current generation
        best_fitness = -fitness_function(population[0])
        best_fitness_values.append(best_fitness)

        # Collision with robots
        for m in range(num_robots):
            for j in range(pop_size):
                for k in range(num_waypoints):
                    for n in range(j + 1, num_robots-1):
                        if (population[m][j][k] == population[m][n][k]).all():
                            continue

        # EnergyConstrain
        Energy_consumed = 10 * best_fitness
        if Energy_consumed > robots_energy:
            continue

        # Print the best fitness value in the current generation
        print(f"Generation {i + 1}: Best fitness = {best_fitness}")

    # Plot the change of the fitness function with respect to the iteration count
    plt.plot(range(1, num_generations + 1), best_fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Change of Fitness with Iteration Count')
    plt.grid(True, alpha=0.4)
    plt.show()

    # Return the best individual as the optimal solution
    return population[0]

# Run the main function and print the optimal solution
solution = main()
print(f"Optimal solution:\n{solution}")

# Define a function to plot the paths of the robots
def plot_paths(paths):
    # Create a new plot with increased size
    plt.figure(figsize=(10, 8))

    # Loop through the number of robots
    for i in range(num_robots):
        # Choose a random color for the robot
        color = np.random.rand(3)

        # Plot the start point as a circle with the same color as the robot
        plt.plot(start_points[i][0], start_points[i][1], 'o', color=color)

        # Plot the goal point as a star with the same color as the robot
        plt.plot(goal_points[i][0], goal_points[i][1], '*', color=color, markersize=10)

        # Include the start point in the waypoints for plotting
        waypoints_for_plotting = np.vstack([start_points[i], paths[i]])

        # Plot the waypoints as a dashed line with the same color as the robot
        plt.plot(waypoints_for_plotting[:, 0], waypoints_for_plotting[:, 1], '--+', color=color, label=f'Robot {i + 1} Path')

    # Plot obstacles if provided
    if obstacle_points is not None:
        plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], marker='x', color='red', label='Obstacles')

    # Set the axis limits
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)

    # Set integer values for x and y-axis ticks
    plt.xticks(np.arange(0, max_x + 3, 3))
    plt.yticks(np.arange(0, max_y + 3, 3))

    # Turn on the grid and set its transparency
    plt.grid(True, alpha=0.4)

    # Set the axis labels
    plt.xlabel("X")
    plt.ylabel("Y")

    # Set the title
    plt.title("Robot Path Planning using Genetic Algorithm")

    # Add a legend to the plot
    plt.legend()

    # Show the graph
    plt.show()

# Call the plot_paths function with the optimal solution
plot_paths(solution)

end_time = time.time()

print("Time taken: ", end_time - start_time, "seconds")