# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    "If the packman has not eaten the power pellet, he will consider the ghost"
    if newScaredTimes[0] == 0:
        "Check if packman is near to the ghost, we will return very negtive value"
        for ghostState in newGhostStates:
            if manhattanDistance(newPos,ghostState.getPosition()) <= 1:
                return -999999
    "We check if there is food near by"
    if newFood.count() < currentGameState.getFood().count():
        return 999999
    "If there is no food near by, we will consider the closest food"
    foodToPacManMDList = [manhattanDistance(newPos,foodPosition) for foodPosition in newFood.asList()]
    "Get the smallest distance"
    minMD = min(foodToPacManMDList)
    return 100/minMD
    
def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def MiniMaxDecision(self,gameState,depth):
      return self.MaxValue(gameState,depth)[1]
  
  def MaxValue(self,gameState,depth):
      "test if game is in terminated state"
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),Directions.STOP]
      "Initialize valueAndAction"
      valueAndAction = [-999999,Directions.STOP]
      "get pacman's action Not include stop"
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      "get the game state after pacman moves"
      pacmanSuccessorGameStateList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      depth -= 1
      for gameStateindex in range(0,len(pacmanSuccessorGameStateList)):
          tmpValue = self.MinValue(pacmanSuccessorGameStateList[gameStateindex],depth)
          "If we found some value is smaller, we will pick the smaller one"
          if valueAndAction[0] < tmpValue:
              valueAndAction = [tmpValue,pacmanActionList[gameStateindex]]    
      return valueAndAction
  
  def MinValue(self,gameState,depth):
      "test if game is in terminated state"
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      value = 999999
      "store the final game state after these ghosts move"
      allGhostGameStates = []
      "get the game state after ghosts move"
      for ghostIndex in range(1,gameState.getNumAgents()):
          "if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move."
          if gameState.isLose() or gameState.isWin():
              allGhostGameStates.append(gameState)
              continue;
          ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
          allGhostGameStates += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
      "Iterate all states to find the min value"
      for gameState in allGhostGameStates:
          tmpValue = self.MaxValue(gameState,depth)
          value = min(value,tmpValue[0])
      return value

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    return self.MiniMaxDecision(gameState,self.depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  "define alpha and beta"
  alpha = -999999
  beta = 999999
  
  def AlphaBetaSearch(self,gameState,depth,alpha,beta):
      return self.MaxValue(gameState,depth,alpha,beta)[1]
  
  def MaxValue(self,gameState,depth,alpha,beta):
      "test if game is in terminated state"
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),Directions.STOP]
      "Initialize valueAndAction"
      valueAndAction = [-999999,Directions.STOP]
      "get pacman's action Not include stop"
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      "get the game state after pacman moves"
      pacmanSuccessorGameStateList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      depth -= 1
      for gameStateindex in range(0,len(pacmanSuccessorGameStateList)):
          tmpValue = self.MinValue(pacmanSuccessorGameStateList[gameStateindex],depth,alpha,beta)
          "If we found some value is smaller, we will pick the smaller one"
          if valueAndAction[0] < tmpValue:
              valueAndAction = [tmpValue,pacmanActionList[gameStateindex]]
          "alpha beta pruning"
          if valueAndAction[0] >= beta:
                return valueAndAction
          alpha = max(alpha,valueAndAction[0])
      return valueAndAction
  
  def MinValue(self,gameState,depth,alpha,beta):
      "test if game is in terminated state"
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      value = 999999
      "store the final game state after these ghosts move"
      allGhostGameStates = []
      "get the game state after ghosts move"
      for ghostIndex in range(1,gameState.getNumAgents()):
          "if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move."
          if gameState.isLose() or gameState.isWin():
              allGhostGameStates.append(gameState)
              continue;
          ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
          allGhostGameStates += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
      "Iterate all states to find the min value"
      for gameState in allGhostGameStates:
          tmpValue = self.MaxValue(gameState,depth,alpha,beta)
          value = min(value,tmpValue[0])
          "alpha beta pruning"
          if value <= alpha:
              return value
          beta = min(beta,value)
      return value

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.AlphaBetaSearch(gameState,self.depth,self.alpha,self.beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def ExpectedMiniMaxDecision(self,gameState,depth):
      return self.ExpectedMaxValue(gameState,depth)[1]
  
  def ExpectedMaxValue(self,gameState,depth):
      "test if game is in terminated state"
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),Directions.STOP]
      "Initialize valueAndAction"
      valueAndAction = [-999999,Directions.STOP]
      "get pacman's action Not include stop"
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      "get the game state after pacman moves"
      pacmanSuccessorGameStateList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      depth -= 1
      for gameStateindex in range(0,len(pacmanSuccessorGameStateList)):
          tmpValue = self.ExpectedMinValue(pacmanSuccessorGameStateList[gameStateindex],depth)
          "If we found some value is smaller, we will pick the smaller one"
          if valueAndAction[0] < tmpValue:
              valueAndAction = [tmpValue,pacmanActionList[gameStateindex]]    
      return valueAndAction
  
  def ExpectedMinValue(self,gameState,depth):
      "test if game is in terminated state"
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      value = 0
      "store the final game state after these ghosts move"
      allGhostGameStates = []
      "get the game state after ghosts move"
      for ghostIndex in range(1,gameState.getNumAgents()):
          "if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move."
          if gameState.isLose() or gameState.isWin():
              allGhostGameStates.append(gameState)
              continue;
          ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
          allGhostGameStates += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
      "Iterate all states to find the min value"
      for gameState in allGhostGameStates:
          tmpValue = self.ExpectedMaxValue(gameState,depth)
          value += tmpValue[0]
      return value / len(allGhostGameStates)
  
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.ExpectedMiniMaxDecision(gameState,self.depth)

def minMSTDistanceToGoal(pacmanPosition, foodLeftList):
    "For this problem we want to compute the MST for pacman to eat all the dots"
    "1. The pacman eat the closest dot first"
    "When we already find the goal"
    if len(foodLeftList) == 0:
        return 0.1
    "Calculate the closest food distance to packman"
    distanceList = [util.manhattanDistance(pacmanPosition,foodPosition) for foodPosition in foodLeftList]
    closestToPacman = min(distanceList)
    "Only one food left"
    if (len(foodLeftList) == 1):
        return closestToPacman
    "2. Next we use prims algorithm to compute MST"
    "Define a heap for prims algorithm , the key for heap is closest manhattanDistance to discovered set"
    import heapq
    h = []
    "Pop the root position from the list"
    closestToPacmanIndexInList = distanceList.index(closestToPacman)
    closestToPacmanFoodPosition = foodLeftList[closestToPacmanIndexInList]
    foodLeftList.pop(closestToPacmanIndexInList)
    "Define MST"
    MSTDistance = closestToPacman
    "Insert root into heap"
    h.append([0,closestToPacmanFoodPosition])
    "Insert other node vertex into heap , key is infinate first"
    for foodPosition in foodLeftList:
        h.append([999999,foodPosition])
    while(len(h) > 0):
        "Once we extract an vertex from heap, we have its shortest distance"
        u = heapq.heappop(h)
        MSTDistance += u[0]
        for v in h:
            if util.manhattanDistance(u[1],v[1]) < v[0]:
                v[0] = util.manhattanDistance(u[1],v[1])
        "After the decrease key operation, we have to maintain heap invariant"
        heapq.heapify(h)
    "total estimation"
    return MSTDistance

def minDistanceToNonScareGhost(pacmanPosition, ghostPositionList, newScaredTimes):
    if allGhostAreScared(newScaredTimes):
        return 999999
    distanceList = []
    closestToPacman = 0.1
    "Calculate the closest non scared ghost distance to packman"
    for ghostIndex in range(0,len(ghostPositionList)):
        if newScaredTimes[ghostIndex] == 0:
            distanceList.append(util.manhattanDistance(pacmanPosition,ghostPositionList[ghostIndex]))
    closestToPacman += min(distanceList)
    return closestToPacman

def minDistanceToScareGhost(pacmanPosition, ghostPositionList, newScaredTimes):
    if noneOfGhostIsScared(newScaredTimes):
        return 999999
    distanceList = []
    closestToPacman = 0.1
    "Calculate the closest non scared ghost distance to packman"
    for ghostIndex in range(0,len(ghostPositionList)):
        if newScaredTimes[ghostIndex] > 0:
            distanceList.append(util.manhattanDistance(pacmanPosition,ghostPositionList[ghostIndex]))
    closestToPacman += min(distanceList)
    return closestToPacman

def minDistanceToCapsules(pacmanPosition, capsulePositionList):
    if len(capsulePositionList) == 0:
        return 999999
    closestToPacman = 0.1
    "Calculate the closest food distance to packman"
    distanceList = [util.manhattanDistance(pacmanPosition,capsulePosition) for capsulePosition in capsulePositionList]
    closestToPacman += min(distanceList)
    return closestToPacman

def anyGhostIsScared(newScaredTimes):
    for ghostScaredTime in newScaredTimes:
        if ghostScaredTime > 0:
            return True
    return False

def noneOfGhostIsScared(newScaredTimes):
    for ghostScaredTime in newScaredTimes:
        if ghostScaredTime > 0:
            return False
    return True

def allGhostAreScared(newScaredTimes):
    for ghostScaredTime in newScaredTimes:
        if ghostScaredTime == 0:
            return False
    return True
     
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I just did the linear combanation of all factors that need to be considered
  """
  "*** YOUR CODE HERE ***"
  ghostStates = [gameState for gameState in currentGameState.getGhostStates()]
  ghostPositionList = [gameState.getPosition() for gameState in ghostStates]
  foodList = currentGameState.getFood().asList()
  pacmanPosition = currentGameState.getPacmanPosition()
  newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  capsulePositionList = currentGameState.getCapsules()
  currentScore = currentGameState.getScore()
  powerPelletFactor = 0
  "get MST distance to food"
  distToFood = minMSTDistanceToGoal(pacmanPosition,foodList)
  "1. If one ghost is scared, try to eat the ghost first, but also we should avoid non scared ghost"
  if anyGhostIsScared(newScaredTimes):
      distToScaredGhost = minDistanceToScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
      distToNonScaredGhost = minDistanceToNonScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
      if len(foodList) <= 2:
          return currentScore + 200/distToFood + 200/distToScaredGhost - 50/distToNonScaredGhost
      else:
          return currentScore + 1000/distToFood + 200/distToScaredGhost - 50/distToNonScaredGhost
  "2. If the ghost is not scared, we use different approach, eat capsule first"
  "get closet closet distance to non scare ghost"
  if noneOfGhostIsScared(newScaredTimes):
       distToNonScaredGhost = minDistanceToNonScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
       distanceToCapsules = minDistanceToCapsules(pacmanPosition,capsulePositionList)
       return currentScore + 1000/distToFood - 50/distToNonScaredGhost + 100/distanceToCapsules
      
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
  depth = 5
  alpha = -999999
  beta = 999999
  
  def AlphaBetaSearch(self,gameState,depth,alpha,beta):
      return self.MaxValue(gameState,depth,alpha,beta)[1]
  
  def MaxValue(self,gameState,depth,alpha,beta):
      "test if game is in terminated state"
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunctionForContest(gameState),Directions.STOP]
      "Initialize valueAndAction"
      valueAndAction = [-999999,Directions.STOP]
      "get pacman's action Not include stop"
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      "get the game state after pacman moves"
      pacmanSuccessorGameStateList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      depth -= 1
      for gameStateindex in range(0,len(pacmanSuccessorGameStateList)):
          tmpValue = self.MinValue(pacmanSuccessorGameStateList[gameStateindex],depth,alpha,beta)
          "If we found some value is smaller, we will pick the smaller one"
          if valueAndAction[0] < tmpValue:
              valueAndAction = [tmpValue,pacmanActionList[gameStateindex]]
          "alpha beta pruning"
          if valueAndAction[0] >= beta:
                return valueAndAction
          alpha = max(alpha,valueAndAction[0])
      return valueAndAction
  
  def MinValue(self,gameState,depth,alpha,beta):
      "test if game is in terminated state"
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunctionForContest(gameState)
      value = 999999
      "store the final game state after these ghosts move"
      allGhostGameStates = []
      "get the game state after ghosts move"
      for ghostIndex in range(1,gameState.getNumAgents()):
          "if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move."
          if gameState.isLose() or gameState.isWin():
              allGhostGameStates.append(gameState)
              continue;
          ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
          allGhostGameStates += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
      "Iterate all states to find the min value"
      for gameState in allGhostGameStates:
          tmpValue = self.MaxValue(gameState,depth,alpha,beta)
          value = min(value,tmpValue[0])
          "alpha beta pruning"
          if value <= alpha:
              return value
          beta = min(beta,value)
      return value
 
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    return self.AlphaBetaSearch(gameState,self.depth,self.alpha,self.beta)

  def evaluationFunctionForContest(self,currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    
      DESCRIPTION: I just did the linear combanation of all factors that need to be considered
    """
    "*** YOUR CODE HERE ***"
    ghostStates = [gameState for gameState in currentGameState.getGhostStates()]
    ghostPositionList = [gameState.getPosition() for gameState in ghostStates]
    foodList = currentGameState.getFood().asList()
    pacmanPosition = currentGameState.getPacmanPosition()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsulePositionList = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()
    powerPelletFactor = 0
    "get MST distance to food"
    distToFood = minMSTDistanceToGoal(pacmanPosition,foodList)
    "1. If one ghost is scared, try to eat the ghost first, but also we should avoid non scared ghost"
    if anyGhostIsScared(newScaredTimes):
        distToScaredGhost = minDistanceToScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
        distToNonScaredGhost = minDistanceToNonScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
        distanceToCapsules = minDistanceToCapsules(pacmanPosition,capsulePositionList)
        "we don't want pacman eat multiple capsules at the same time"
        if pacmanPosition in capsulePositionList:
            return -999999
        value = currentScore + 1000/distToFood + 200/distToScaredGhost - 50/distToNonScaredGhost
        return value
    "2. If the ghost is not scared, we use different approach, eat capsule first"
    "get closet closet distance to non scare ghost"
    if noneOfGhostIsScared(newScaredTimes):
         distToNonScaredGhost = minDistanceToNonScareGhost(pacmanPosition, ghostPositionList, newScaredTimes)
         distanceToCapsules = minDistanceToCapsules(pacmanPosition,capsulePositionList)
         value = currentScore + 1000/distToFood - 50/distToNonScaredGhost + 100/distanceToCapsules
         return value
