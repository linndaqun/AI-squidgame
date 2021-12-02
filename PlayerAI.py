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

DEPTH_LIMIT = 5

# TO BE IMPLEMENTED
# 
class PlayerAI(BaseAI):

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
        """ Returns a random, valid move """
        
        # find all available moves 
        available_moves = grid.get_neighbors(self.pos, only_available = True)

        # make random move
        new_pos = random.choice(available_moves) if available_moves else None
        
        return new_pos

    def getTrap(self, grid : Grid, weight=[1,1,1]) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        
        # TODO: cool edge case to implement
        opponent_pos = grid.find(3-self.player_num)
        if len(grid.get_neighbors(opponent_pos, only_available=True)) == 0:
            return None

        # throw to one of the available cells based on heuristics
        (trap, _) = self.trap_maximize(grid, self.player_num, 0, inf, -inf, weight)

        return trap
    
    def trap_maximize(self, grid, player, depth, alpha, beta, weight) -> tuple:
        '''
            Find the child state with the highest utility value

            This function should return a tuple (maxTrap, maxUtil), where maxTrap is the cell with the highest 
            utility value and maxUtil is the value
        '''

        #find position
        opponent_pos = grid.find(3-player)
        player_pos = grid.find(player)

        # terminal test when the depth limit is reached
        if depth == DEPTH_LIMIT:
            util = 0
            util += weight[0] * heuristic_IS(grid, opponent_pos, player_pos)
            util += weight[1] * heuristic_AIS(grid, opponent_pos, player_pos)
            util += weight[2] * heuristic_OCLS(grid, opponent_pos, player_pos)
            return (None, util)
        
        (maxTrap, maxUtil) = (None, -inf)
        # find all available cells surrounding Opponent
        available_cells = grid.get_neighbors(opponent_pos, only_available=True)
        for cell in available_cells:
            new_grid = grid.clone()
            new_grid.trap(cell)
            (_, util) = self.move_minimize(new_grid, 3-player, depth+1, alpha, beta, weight)
            if util > maxUtil:
                maxUtil = util
                maxTrap = cell
            if maxUtil >= beta:
                break
            if maxUtil > alpha:
                alpha = maxUtil
        return (maxTrap, maxUtil)

    def move_minimize(self, grid, player, depth, alpha, beta, weight) -> tuple:
        '''
                Find the child state with the lowest utility value

                This function should return a tuple (minMove, minUtil), where minMove is the cell with the lowest
                utility value and minUtil is the value
        '''

        opponent_pos = grid.find(3-player)
        player_pos = grid.find(player)

        # terminal test when the depth limit is reached
        if depth == DEPTH_LIMIT:
            util = 0
            util += weight[0] * heuristic_IS(grid, opponent_pos, player_pos)
            util += weight[1] * heuristic_AIS(grid, opponent_pos, player_pos)
            util += weight[2] * heuristic_OCLS(grid, opponent_pos, player_pos)
            return (None, util)
        
        (minMove, minUtil) = (None, inf)
        # find all available cells surrounding Opponent
        available_cells = grid.get_neighbors(player_pos, only_available=True)
        for cell in available_cells:
            new_grid = grid.clone()
            new_grid.move(cell, player)
            (_, util) = self.trap_maximize(new_grid, 3-player, depth+1, alpha, beta, weight)
            if util < minUtil:
                minUtil = util
                minMove = cell
            if minUtil <= alpha:
                break
            if minUtil < beta:
                beta = minUtil
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