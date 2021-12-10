from math import inf
import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
from Displayer import Displayer
from test_players.opponent_heuristics import heuristic_IS
from Utils import *

DEPTH_LIMIT = 5

# TO BE IMPLEMENTED
# 
class Opponent_minimax(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        available_moves = grid.get_neighbors(self.pos, only_available=True)
        heuristic_scores = []

        for next_move in available_moves:
            next_state = grid.clone().move(next_move, self.player_num)
            heuristic_score = MoveMinimax(next_state, 1, self.player_num, False)
            heuristic_scores.append(heuristic_score)

        return available_moves[np.argmax(heuristic_scores)]

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        
        # TODO: cool edge case to implement

        # throw to one of the available cells based on heuristics
        (trap, _) = self.trap_maximize(grid, self.player_num, 0)

        return trap
    
    def trap_maximize(self, grid, player, depth) -> tuple:
        '''
            Find the child state with the highest utility value

            This function should return a tuple (maxTrap, maxUtil), where maxTrap is the cell with the highest 
            utility value and maxUtil is the value
        '''
        # choose heuristics
        heuristic_func = heuristic_IS

        #find position
        opponent_pos = grid.find(3-player)
        player_pos = grid.find(player)

        # terminal test when the depth limit is reached
        if depth == DEPTH_LIMIT:
            util = heuristic_func(grid, opponent_pos, player_pos)
            p = 1 - 0.05*(manhattan_distance(player_pos, opponent_pos) - 1)
            return (None, util)
        
        (maxTrap, maxUtil) = (None, -inf)
        # find all available cells surrounding Opponent
        available_cells = grid.get_neighbors(opponent_pos, only_available=True)
        for cell in available_cells:
            new_grid = grid.clone()
            new_grid.trap(cell)
            (_, util) = self.move_minimize(new_grid, 3-player, depth+1)
            if util > maxUtil:
                maxUtil = util
                maxTrap = cell
        return (maxTrap, maxUtil)

    def move_minimize(self, grid, player, depth) -> tuple:
        '''
                Find the child state with the lowest utility value

                This function should return a tuple (minMove, minUtil), where minMove is the cell with the lowest
                utility value and minUtil is the value
        '''
        # choose heuristics
        heuristic_func = heuristic_IS

        opponent_pos = grid.find(3-player)
        player_pos = grid.find(player)

        # terminal test when the depth limit is reached
        if depth == DEPTH_LIMIT:
            util = heuristic_func(grid, opponent_pos, player_pos)
            return (None, util)
        
        (minMove, minUtil) = (None, inf)
        # find all available cells surrounding Opponent
        available_cells = grid.get_neighbors(player_pos, only_available=True)
        for cell in available_cells:
            new_grid = grid.clone()
            new_grid.move(cell, player)
            (_, util) = self.trap_maximize(new_grid, 3-player, depth+1)
            if util < minUtil:
                minUtil = util
                minMove = cell
        return (minMove, minUtil)

        
def heuristic_IS(grid, opponent, player):
    oppo_cells = grid.get_neighbors(opponent, only_available=True)
    player_cells = grid.get_neighbors(player, only_available=True)
    return len(player_cells) - len(oppo_cells)

def heuristic_AIS(grid, opponent, player):
    oppo_cells = grid.get_neighbors(opponent, only_available=True)
    player_cells = grid.get_neighbors(player, only_available=True)
    return len(player_cells) - 0.5 * len(oppo_cells)

def heuristic_OCLS(grid, opponent, player):
    oppo_moves, player_moves = 0, 0
    oppo_cells = grid.get_neighbors(opponent, only_available=True)
    for cell in oppo_cells:
        oppo_moves += len(grid.get_neighbors(cell, only_available=True))
    player_cells = grid.get_neighbors(player, only_available=True)
    for cell in player_cells:
        player_moves += len(grid.get_neighbors(cell, only_available=True))
    return player_moves - oppo_moves

# The ExpectMinimax algorithm of Move
# Parameters:
def MoveMinimax(grid: Grid, cur_depth, player_num, player_turn):
    if cur_depth == 5 or not grid.get_neighbors(grid.find(player_num), only_available=True):
        return getHeuristic(grid, player_num)
    if player_turn:
        available_moves = grid.get_neighbors(grid.find(player_num), only_available=True)
        next_states = [grid.clone().move(mv, player_num) for mv in available_moves]
        heuristic_scores = [MoveMinimax(next_state, cur_depth + 1, player_num, not player_turn) for next_state in next_states]
        return np.argmax(heuristic_scores)
    else:
        available_traps = grid.get_neighbors(grid.find(player_num), only_available=True)
        next_states = [grid.clone().trap(trap) for trap in available_traps]
        heuristic_scores = [MoveMinimax(next_state, cur_depth + 1, player_num, not player_turn) for next_state in next_states]

        opponent_pos = grid.find(3 - player_num)
        # calculates probability of landing at each available cell
        for index, trap in enumerate(available_traps):
            manhattan_distance = abs(opponent_pos[0] - trap[0]) + abs(opponent_pos[1] - trap[1])
            probability = 1 - 0.05 * (manhattan_distance - 1)
            heuristic_scores[index] *= probability

        return np.argmin(heuristic_scores)

# Heuristic Function: the difference between the current number of moves Player (You) can make
# and the current number of moves the opponent can make.
def getHeuristic(grid: Grid, player_num):
    available_moves = grid.get_neighbors(grid.find(player_num), only_available=True)
    num_available_moves = len(available_moves)

    opponent_pos = grid.find(3 - player_num)
    opponent_available_moves = grid.get_neighbors(opponent_pos, only_available=True)
    opponent_num_available_moves = len(opponent_available_moves)

    return num_available_moves - opponent_num_available_moves

# Given the current player, retrieves all the available traps (available neighboring cells)
# Returns a list of (trap_pos, probability)
def getAvailableTraps(grid: Grid, player_num):
    available_traps = np.empty
    available_cells = grid.get_neighbors(grid.find(player_num), only_available=True)

    opponent_player_num = 3 - player_num
    opponent_pos = grid.find(opponent_player_num)

    # opponent will throw the trap at the cell that maximizes his score
    for trap in available_cells:
        next_state = grid.clone().trap(trap)
        heuristic_score = getHeuristic(next_state, opponent_player_num)
        manhattan_distance = abs(opponent_pos[0] - trap[0]) + abs(opponent_pos[1] - trap[1])
        probability = 1 - 0.05 * (manhattan_distance - 1)


    return available_traps


# Returns the state after the opponent selects the optimal cell to trap
def getOpponentTrap(grid: Grid, player_num):
    available_cells = grid.get_neighbors(grid.find(player_num), only_available=True)
    heuristic_scores = []

    opponent_player_num = 3 - player_num
    opponent_pos = grid.find(opponent_player_num)

    # opponent will throw the trap at the cell that maximizes his score
    for trap in available_cells:
        next_state = grid.clone().trap(trap)
        heuristic_score = getHeuristic(next_state, opponent_player_num)
        manhattan_distance = abs(opponent_pos[0] - trap[0]) + abs(opponent_pos[1] - trap[1])
        probability = 1 - 0.05 * (manhattan_distance - 1)
        heuristic_scores.append(heuristic_score * probability)

    opponent_selected_trap = available_cells[np.argmax(heuristic_scores)]
    next_state = grid.clone().trap(opponent_selected_trap)

    return next_state