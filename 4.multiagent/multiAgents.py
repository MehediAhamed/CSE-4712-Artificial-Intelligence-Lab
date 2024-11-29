# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        closestFoodDistance, closestGhostDistance= float("inf"), float("inf")
        foodDistances, ghostDistances = [closestFoodDistance], [closestGhostDistance]
        foodList = newFood.asList()                                     # after successor action foodlist
        currentFoodNumber = len(currentGameState.getFood().asList())    # current foodlist

        for food in foodList:
            foodDistances.append(util.manhattanDistance(newPos, food))
        closestFoodDistance = min(foodDistances)
        
        for ghost in newGhostStates:
            ghostDistances.append(util.manhattanDistance(newPos, ghost.getPosition()))
        closestGhostDistance = min(ghostDistances)
        
        foodDiff = currentFoodNumber - len(foodList)
        evalScore = 1/(closestFoodDistance + 1 +  closestGhostDistance)          # closer to food and further away from ghost the better

        if closestGhostDistance <= 1:                                            # ghost eat pacman so high penalty
            return -float("inf")
        elif foodDiff == 1:                                                      # foodDiff 1 means pacman ate a food
            return float("inf")
        else:
            return evalScore

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        optimalValue, optimalAction = self.value(gameState, 0, self.depth)
        return optimalAction

    def value(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def minValue(self, gameState, agentIndex, depth):
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth -= 1

        legalMoves = gameState.getLegalActions(agentIndex)
        stateValue, stateAction = 1e9, None

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue, successorAction = self.value(successorGameState, nextAgentIndex, nextDepth)
            if successorValue < stateValue:
                stateValue = successorValue
                stateAction = action

        return stateValue, stateAction

    def maxValue(self, gameState, agentIndex, depth):
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth -= 1

        legalMoves = gameState.getLegalActions(agentIndex)
        stateValue, stateAction = -1e9, None

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue, successorAction = self.value(successorGameState, nextAgentIndex, nextDepth)
            if successorValue > stateValue:
                stateValue = successorValue
                stateAction = action

        return stateValue, stateAction
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        optimalValue, optimalAction = self.value(gameState, 0, self.depth, -float("inf"), float("inf"))
        return optimalAction
    
    def value(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == 0:               
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)    # Pacman is maxValue
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)    # Ghosts are minValue
   
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalMoves = gameState.getLegalActions(agentIndex)
        currentScore, currentAction = float("inf"), Directions.STOP

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue, successorAction = self.value(successorGameState, nextAgent, nextDepth, alpha, beta)

            if currentScore > successorValue:
                currentScore = successorValue
                currentAction = action
            # Pruning
            if currentScore < alpha:
                return currentScore, currentAction
            beta = min(beta, currentScore)

        return currentScore, currentAction
    
    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        legalMoves = gameState.getLegalActions(agentIndex)
        currentScore, currentAction = -float("inf"), Directions.STOP

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.value(successorGameState, nextAgent, nextDepth, alpha, beta)[0]
            
            if currentScore < successorValue:
                currentScore = successorValue
                currentAction = action
            # Pruning
            if currentScore > beta:
                return currentScore, currentAction
            alpha = max(alpha, currentScore)

        return currentScore, currentAction 
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        optimalValue, optimalAction= self.value(gameState, 0, self.depth)
        return optimalAction
    

    def expval(self, gameState, agentIndex, depth):
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        expScore, prob = 0, 0
        legalMoves = gameState.getLegalActions(agentIndex)


        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue, successorAction = self.value(successorGameState, nextAgent, nextDepth)
            
            prob = 1 / len(legalMoves) 
            expScore += prob * successorValue
            
        return expScore, None
    

    def value(self, gameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth== 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex==0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expval(gameState,agentIndex, depth)
    def maxValue(self, gameState, agentIndex, depth):
        nextAgent = (agentIndex + 1) % gameState.getNumAgents() 
        if nextAgent == 0:
            nextDepth = depth - 1 
        else:
            nextDepth = depth

        currentScore, currentAction = -1e9, Directions.STOP 
        legalMoves = gameState.getLegalActions(agentIndex) 

        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.value(successorGameState, nextAgent, nextDepth)[0] 
            if currentScore < successorValue:
                currentScore = successorValue 
                currentAction = action 

        return currentScore, currentAction 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScore = currentGameState.getScore()
    currCapsList = currentGameState.getCapsules()
    
    currFoodList = currFood.asList()

    closestFoodDistance, closestGhostDistance, closestCapsuleDistance = 1e9, 1e9, 1e9
    foodDistances, ghostDistances, capsuleDistances = [closestFoodDistance], [closestGhostDistance], [closestCapsuleDistance]

    foodCoeff = 1
    scoreCoeff = 1
    
    capsuleCoeff = 3                
    ghostDistCoeff = 2              
    scaredGhostTimer = 4


    # closest food 
    for food in currFoodList:
        foodDistances.append(util.manhattanDistance(currPos, food))
    closestFoodDistance = min(foodDistances)

    # closest capsule
    for capsule in currCapsList:
        capsuleDistances.append(util.manhattanDistance(currPos, capsule))
    closestCapsuleDistance = min(capsuleDistances)

    # dealing with ghost  
    for ghost in currGhostStates:
        ghostDistance = util.manhattanDistance(currPos, ghost.getPosition())
        if ghostDistance <= ghostDistCoeff and ghost.scaredTimer > scaredGhostTimer:
            return 1e9
        elif ghostDistance <= ghostDistCoeff and ghost.scaredTimer <= scaredGhostTimer:
            return -1e9
            
    evalScore = foodCoeff * (1/(closestFoodDistance + 1)) + capsuleCoeff * 1/(closestCapsuleDistance + 1) + scoreCoeff * currScore
    return  evalScore
  

# Abbreviation
better = betterEvaluationFunction
