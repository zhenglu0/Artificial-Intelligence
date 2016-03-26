# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    "Check if start state is the solution"
    if problem.isGoalState(problem.getStartState()):
        return []
    "Initialize virables"
    frontier.push(problem.getStartState())
    frontierDict = {problem.getStartState():[]}
    explored = set()
    "Begin extraction node from queue"
    while not frontier.isEmpty():
        currentState = frontier.pop()
        explored.add(currentState)
        "Get the current path list"
        currentPath = frontierDict[currentState]
        del frontierDict[currentState]
        "Get the successors of the current node"
        successors = problem.getSuccessors(currentState)
        for node in successors:
            if node[0] not in frontierDict and node[0] not in explored:
                "Copy the previous path that lead to the node"
                nodePath = currentPath[:]
                "Update the path of the node"
                nodePath.append(node[1])
                "Add node to the dictionary"
                frontierDict[node[0]] = nodePath
                "Check the goal state"
                if problem.isGoalState(node[0]):
                    return nodePath
                frontier.push(node[0])
    "When every possible node has been explored and no solution found return empty list"
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    "Check if start state is the solution"
    if problem.isGoalState(problem.getStartState()):
        return []
    "Initialize virables"
    frontier.push(problem.getStartState())
    frontierDict = {problem.getStartState():[]}
    explored = set()
    "Begin extraction node from queue"
    while not frontier.isEmpty():
        currentState = frontier.pop()
        explored.add(currentState)
        "Get the current path list"
        currentPath = frontierDict[currentState]
        del frontierDict[currentState]
        "Get the successors of the current node"
        successors = problem.getSuccessors(currentState)
        for node in successors:
            if node[0] not in frontierDict and node[0] not in explored:
                "Copy the previous path that lead to the node"
                nodePath = currentPath[:]
                "Update the path of the node"
                nodePath.append(node[1])
                "Add node to the dictionary"
                frontierDict[node[0]] = nodePath
                "Check the goal state"
                if problem.isGoalState(node[0]):
                    return nodePath
                frontier.push(node[0])
    "When every possible node has been explored and no solution found return empty list"
    return []

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    "Check if start state is the solution"
    if problem.isGoalState(problem.getStartState()):
        return []
    "Initialize virables"
    frontier.push(problem.getStartState(),0)
    frontierDict = {problem.getStartState():(0,[])}
    explored = set()
    "Begin extraction node from queue"
    while not frontier.isEmpty():
        currentState = frontier.pop()
        explored.add(currentState)
        "Get the current path list"
        currentPath = frontierDict[currentState][1]
        "Get the current path cost"
        currentCost = frontierDict[currentState][0]
        "Check the goal state"
        if problem.isGoalState(currentState):
            return currentPath
        "Get the successors of the current node"
        successors = problem.getSuccessors(currentState)
        for node in successors:
            "Calculate node cost"
            nodeCost = currentCost+node[2]
            if (node[0] not in explored and node[0] not in frontierDict) or (node[0] in frontierDict and nodeCost < frontierDict[node[0]][0]):
                "Copy the previous path that lead to the node"
                nodePath = currentPath[:]
                "Update the path of the node"
                nodePath.append(node[1])
                "Add node to the explore"
                frontierDict[node[0]] = (nodeCost,nodePath)
                "Insert into the queue"
                frontier.push(node[0],nodeCost)
    "When every possible node has been explored and no solution found return empty list"
    return []
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    "Check if start state is the solution"
    if problem.isGoalState(problem.getStartState()):
        return []
    "Initialize virables"
    frontier.push(problem.getStartState(),heuristic(problem.getStartState(),problem))
    frontierDict = {problem.getStartState():(0,[])}
    explored = set()
    "Begin extraction node from queue"
    while not frontier.isEmpty():
        currentState = frontier.pop()
        explored.add(currentState)
        "Get the current path list"
        currentPath = frontierDict[currentState][1]
        "Get the current path cost"
        currentCost = frontierDict[currentState][0]
        "Check the goal state"
        if problem.isGoalState(currentState):
            return currentPath
        "Get the successors of the current node"
        successors = problem.getSuccessors(currentState)
        for node in successors:
            "Calculate node cost"
            nodeCost = currentCost+node[2]
            if (node[0] not in explored and node[0] not in frontierDict) or (node[0] in frontierDict and nodeCost < frontierDict[node[0]][0]):
                "Copy the previous path that lead to the node"
                nodePath = currentPath[:]
                "Update the path of the node"
                nodePath.append(node[1])
                "Add node to the explore"
                frontierDict[node[0]] = (nodeCost,nodePath)
                "Insert into the queue"
                frontier.push(node[0],nodeCost+heuristic(node[0],problem))
    "When every possible node has been explored and no solution found return empty list"
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
