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

import matplotlib.pyplot as plt
from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
import time
from game import Agent
from pacman import GameState
import random
from util import manhattanDistance

class ReflexAgent(Agent):
    """
    A reflex agent with randomized tie-breaking that chooses an action at each choice point
    by examining its alternatives via a state evaluation function.
    """
    def __init__(self, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta  # Maximum random jitter for tie-breaking
        self.clear_times = []  # Track clear times across games
        self.current_game_start = None
        self._active_game = False
        
        # New variables for win/loss tracking
        self.total_wins = 0
        self.total_losses = 0

    def registerWin(self):
        self.total_wins += 1
        if self.current_game_start is not None:
            self.clear_times.append(time.time() - self.current_game_start)
            self.current_game_start = None
            self._active_game = False

    def registerLoss(self):
        self.total_losses += 1
        self._active_game = False

    def final(self, gameState):
        if gameState.isLose():
            self.registerLoss()
        
        self._active_game = False

    def getAction(self, gameState):
        """
        Choose among the best options according to evaluation function with random jitter.
        """
        if self.current_game_start is None:
            self.current_game_start = time.time()
        legalMoves = gameState.getLegalActions()
        

        # Evaluate each move with random jitter
        scores = []
        for action in legalMoves:
            baseScore = self.evaluationFunction(gameState, action)
            epsilon = random.uniform(0, self.delta)  # Add jitter between 0 and delta
            scores.append(baseScore + epsilon)
        
        # Find the best score and all actions that achieve it
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        # Randomly choose among the best actions
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def printStats(self):
        if not self.clear_times:
            print("No games won yet!")
            return
    
        print("\n=== Performance Analysis ===")
        print(f"Total games: {self.total_wins + self.total_losses}")
        print(f"Wins: {self.total_wins} | Losses: {self.total_losses}")
        print(f"Win rate: {self.total_wins/(self.total_wins+self.total_losses)*100:.1f}%")
        
        if self.clear_times:
            print("\nTiming Statistics:")
            print(f"Average win time: {np.mean(self.clear_times):.2f} sec")
            print(f"Fastest win: {np.min(self.clear_times):.2f} sec")
            print(f"Slowest win: {np.max(self.clear_times):.2f} sec")
          # Make sure we have game count
        self._plotWinRatio()

    def _plotWinRatio(self):
        plt.figure(figsize=(8, 6))
    
        # Data to plot
        sizes = [self.total_wins, self.total_losses]
        labels = [f'Wins: {self.total_wins}', f'Losses: {self.total_losses}']
        colors = ['#4CAF50', '#F44336']  # Green and red
        explode = (0.05, 0)  # Slightly highlight the wins
    
        # Create pie chart
        plt.pie(sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=140)
    
        # Equal aspect ratio ensures pie is drawn as circle
        plt.axis('equal')
    
        # Add title with proper line continuation
        total_games = self.total_wins + self.total_losses
        title_text = (f'Win/Loss Ratio ({self.total_wins}/{total_games} Games)\n'
                        f'Average Win Time: {np.mean(self.clear_times):.2f} sec')
        plt.title(title_text, pad=20)
    
        plt.tight_layout()
        plt.show()

    def evaluationFunction(self, currentGameState, action):
        """
        Improved evaluation function that considers:
        - Current game score
        - Distance to nearest food
        - Ghost proximity
        - Scared ghosts
        - Number of remaining food dots
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Base score from game state
        score = successorGameState.getScore()

        # Food evaluation
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / (minFoodDist + 1)  # +1 to avoid division by zero

        # Ghost evaluation
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[i] > 0:  # Chase scared ghosts
                if ghostDist > 0:
                    score += 1.0 / ghostDist * 2  # Extra bonus for chasing scared ghosts
            else:  # Avoid normal ghosts
                if ghostDist < 3:
                    score -= 2.0 / (ghostDist + 1)  # Penalty increases as ghost gets closer

        # Bonus for having less food remaining
        score -= len(foodList) * 0.5

        return score


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
        """
        def minimax(state, depth, agentIndex):
            # Base cases
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Pacman's turn (maximizer)
            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            # Ghosts' turn (minimizers)
            else:
                return min_value(state, depth, agentIndex)
        
        def max_value(state, depth, agentIndex):
            v = -float('inf')
            legalActions = state.getLegalActions(agentIndex)
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                # Next agent is first ghost (agentIndex + 1)
                v = max(v, minimax(successor, depth, agentIndex + 1))
            return v
        
        def min_value(state, depth, agentIndex):
            v = float('inf')
            legalActions = state.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            # Check if we need to wrap around to Pacman (agent 0)
            if nextAgent >= state.getNumAgents():
                nextAgent = 0
                depth += 1  # Increase depth after all agents have moved
            
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax(successor, depth, nextAgent))
            return v
        
        # Get the best action for Pacman (agent 0)
        legalActions = gameState.getLegalActions(0)
        scores = []
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # Start with depth 0 and next agent (first ghost)
            scores.append((minimax(successor, 0, 1), action))
        
        # Return action with highest score
        return max(scores)[1]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning
        """
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            value = -float('inf')
            actions = state.getLegalActions(0)
            
            for action in actions:
                successor = state.generateSuccessor(0, action)
                value = max(value, min_value(successor, depth, 1, alpha, beta))
                if value > beta:
                    #If max finds a value ≥ beta → min will avoid this → prune
                    return value
                alpha = max(alpha, value)
            return value
        
        def min_value(state, depth, ghostIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            value = float('inf')
            actions = state.getLegalActions(ghostIndex)
            
            for action in actions:
                successor = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:
                    value = min(value, max_value(successor, depth + 1, alpha, beta))
                else:
                    value = min(value, min_value(successor, depth, ghostIndex + 1, alpha, beta))
                
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
        
        # Main body of getAction
        best_action = None
        alpha = -float('inf')
        beta = float('inf')
        legal_actions = gameState.getLegalActions(0)
        
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = min_value(successor, 0, 1, alpha, beta)
            
            if value > alpha:
                alpha = value
                best_action = action
        
        return best_action
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = -float('inf')
            actions = state.getLegalActions(0)
            
            for action in actions:
                successor = state.generateSuccessor(0, action)
                v = max(v, exp_value(successor, depth, 1))
            return v
        
        def exp_value(state, depth, ghostIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            v = 0
            actions = state.getLegalActions(ghostIndex)
            prob = 1.0 / len(actions)  # Uniform probability
            
            for action in actions:
                successor = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:
                    v += prob * max_value(successor, depth + 1)
                else:
                    v += prob * exp_value(successor, depth, ghostIndex + 1)
            return v
        
        # Main expectimax procedure
        best_action = None
        best_value = -float('inf')
        legal_actions = gameState.getLegalActions(0)
        
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = exp_value(successor, 0, 1)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (question 5).
    
    DESCRIPTION: This evaluation function considers:
    1. Current score
    2. Distance to nearest food
    3. Number of remaining food pellets
    4. Distance to nearest capsule
    5. Number of remaining capsules
    6. Ghost distances (with different weights for scared vs active ghosts)
    7. Whether Pacman is in a "stop" state
    """
    # Extract useful information from the current game state
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    
    # Initialize features
    score = currentGameState.getScore()
    foodDist = []
    ghostDist = []
    capsuleDist = []
    
    # Calculate distances to all food pellets
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                foodDist.append(manhattanDistance(pacmanPosition, (x, y)))
    
    # Calculate distances to ghosts (consider scared time)
    for ghost in ghosts:
        dist = manhattanDistance(pacmanPosition, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Positive value for scared ghosts (we want to chase them)
            ghostDist.append(dist)
        else:
            # Negative value for active ghosts (we want to avoid them)
            ghostDist.append(-dist)
    
    # Calculate distances to capsules
    for capsule in capsules:
        capsuleDist.append(manhattanDistance(pacmanPosition, capsule))
    
    # Feature calculations
    features = {}
    
    # Current game score (already weighted by game rules)
    features['score'] = score
    
    # Food-related features
    if foodDist:
        features['nearestFood'] = 1.0 / min(foodDist) #كل ما يقرب كل ما كان احسن
        features['foodCount'] = -len(foodDist)  # Negative because less food is better
    else:
        features['nearestFood'] = 0
        features['foodCount'] = 0
    
    # Ghost-related features
    if ghostDist:
        # Active ghosts (negative distances)
        activeGhosts = [d for d in ghostDist if d < 0]
        if activeGhosts:
            features['nearestActiveGhost'] = max(activeGhosts)  # Closest is least negative
        
        # Scared ghosts (positive distances)
        scaredGhosts = [d for d in ghostDist if d > 0]
        if scaredGhosts:
            features['nearestScaredGhost'] = min(scaredGhosts)
    else:
        features['nearestActiveGhost'] = 0
        features['nearestScaredGhost'] = 0
    
    # Capsule-related features
    if capsuleDist:
        features['nearestCapsule'] = 1.0 / min(capsuleDist)
        features['capsuleCount'] = -len(capsules)
    else:
        features['nearestCapsule'] = 0
        features['capsuleCount'] = 0
    
    # Weighted sum of features
    weights = {
        'score': 1.0,
        'nearestFood': 10.0,
        'foodCount': -100.0,
        'nearestActiveGhost': 10.0,
        'nearestScaredGhost': -20.0,  # Negative because we want to minimize distance to scared ghosts
        'nearestCapsule': 30.0,
        'capsuleCount': -100.0
    }
    
    # Calculate final evaluation
    evaluation = 0
    for key in features:
        evaluation += features[key] * weights[key]
    
    return evaluation

# Abbreviation
better = betterEvaluationFunction
    # This lets pacman.py detect the agent
