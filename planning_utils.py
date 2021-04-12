from enum import Enum
from queue import PriorityQueue
import numpy as np
import scipy.spatial.distance as dist
import heapdict
import sys

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions

# Given function for A Star Search
def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


# Question 1 - Depth First Search
def dfs(grid, h, start, goal):

    path = []
    path_cost = 0

    #use a list as a stack
    stack = []
    stack.append((0,start))
    visited = set(start)

    #set a depth limit that is large enough - to make the performance better (h(start, goal) was found to be too small for some values of start and goal)
    depth_limit = h(start, goal)**2

    branch = {}
    found = False
    
    while len(stack):

        #removes and returns last item of the list (idx -1)
        item = stack.pop()
        current_node = item[1]
        
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        elif depth_limit>=item[0]:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    #append it to the list
                    stack.append((branch_cost, next_node))     
             
    if found:
        # retrace steps
        print("Retracing steps....")
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost                

#Question 2 - Iterative Deepening A Star
def iterative_astar(grid, h, start, goal):
    path = []
    path.append(start)
    branch = {}
    thd = h(start,goal) #initiate threshold value
    found = False
    while True:
        gScore = 0.0    #initial gScore is 0 since there is no parent node at start node
        temporaryThd = searchMinFScore(path, gScore, h, grid, thd, goal, branch) 
        if temporaryThd < 0:
            found = True
            print('Found a path.')
            break
        if temporaryThd == float('inf'): #infinity in python
            break
        thd = temporaryThd
    if found:
        n = goal
        path_cost = branch[n][0]
            
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
        
    return path[::-1], path_cost
    
    
#helper function for Iterative Deepening A Star
def searchMinFScore(path, gScore, h, grid, thd, goal, branch):
    
    minFScore = sys.maxsize * 2 + 1 #initialize as a really large value in python - to be updated later
    current_node = path[-1] # get the last node in the path
    fScore = gScore + h(current_node,goal)

    if fScore>thd:
        return fScore
    if current_node == goal:
        return -fScore # a negative return value
    
    for action in valid_actions(grid, current_node):
        da = action.delta 
        next_node = (current_node[0]+da[0],current_node[1]+da[1])
        if next_node not in path:
            path.append(next_node)
            branch_cost = gScore + action.cost
            branch[next_node] = (branch_cost,current_node,action)
            temp = searchMinFScore(path,branch_cost,h,grid,thd,goal,branch)
            if temp < 0:
                return temp
            if(temp<minFScore):
                minFScore = temp
            path.pop()
            branch.pop(next_node)

    return minFScore 

#Question 3 - Uniform Cost Search
def ucs(grid, h, start, goal):

    path = []
    path_cost = 0
    heapDictionary = heapdict.heapdict() # justification: heap dictionary is a way to combine the logics of data structures priority queue and key-value pair logic of dictionary
    heapDictionary[start] = 0
    visited = set(start)
    branch = {}
    found = False
    
    while heapDictionary:
        item = heapDictionary.popitem()
        current_node = item[0]
        if current_node == start:
            current_cost = 0.0
        else: 
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    heapDictionary[next_node] = branch_cost

                elif next_node in heapDictionary.keys() and branch_cost < heapDictionary[next_node]:       #found a better (lower cost) path to a node => update the stored cost and the path
                    print("node encountered again....path is better...cost"+branch_cost+"<"+heapDictionary[next_node])
                    heapDictionary[next_node] = branch_cost
                    branch[next_node] = (branch_cost, current_node, action)
                
                if(next_node==goal):
                    print("A path to goal found with cost={}".format(branch_cost))

             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

#Question 4 - Heuristic Functions

#Heuristic uses Euclidian Distance
def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

#Heuristic uses Manhattan Distance
# Using a_star() with Manhattan heuristic, it travels a path with an "L" shape (+23, -19)
def heuristicManhattanForQ4(position, goal_position):
    return dist.cityblock(position, goal_position)
    # return abs(position[0]-goal_position[0]) + abs(position[1]-goal_position[1])

#Heuristic uses Minkowski Distance with p=3
# Using a_star() with Minkowski (p=3) heuristic, it travels a path with straight and braided path (+23, -19)
def heuristicMinkowskiForQ4(position, goal_position):
    return dist.minkowski(position, goal_position, p=3)

#Question 6 - Pass through 3 points A Star Search
def threePointsAStar(grid, grid_start, grid_goal):
    path = []
    pts = getThreePoints()
    path_cost = 0 
    current_start = grid_start
    while len(pts):
        index = getClosestPt(current_start,pts)
        temp,tempCost = a_star(grid,heuristic,current_start,pts[index])
        path =  path + temp
        path_cost = path_cost+tempCost
        print('\nCurrent Path is = {}'.format(temp))
        current_start = pts[index]
        pts.pop(index)
    
    final_path,finalCost = a_star(grid,heuristic,current_start,grid_goal)
    path = path + final_path
    path_cost = path_cost+ finalCost
    print('\nFinal Path is = {}'.format(path))
    return path, path_cost

#sets the three points to pass through
def getThreePoints():
    pts = []
    pts.append((317,450))
    pts.append((313,451))
    pts.append((323,430))
    return pts

#gets the point that is the closest to the current point
def getClosestPt(start, pts):
    minimum = heuristic(start, pts[0])
    index = 0
    for p in pts:
        if minimum>heuristic(start, p):
            minimum = heuristic(start, p)
            index = pts.index(p)
    print('\n The current minimum is {} for index {} of pts'.format(minimum,index))
    return index


