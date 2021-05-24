% BLOCKS WORLD
% A cannonical planning problem

% PROBLEM-SPECIFIC DOMAIN (UNINTERPRETED FUNCTIONS) ****************************

%	VARIABLES:
%	B			... describes a block (`b1`, `b2`, ...) in the blocks world.
%	L			... describes a location to put blocks on top of
%					(`table`, `b1`, `b2`, ...).

% 	STATE FUNCTIONS S(...):
%	on(B,L)		... describes block `B` being stacked on top of location `L`.
%	goal		... labels states in which all subgoals are reached.

%	ACTION FUNCTIONS A(...):
%	move(B,L)	... describes the action of moving block `B` to location `L`.

%	STATIC STATE (ATOMS):
%	subgoal(B,L)	... describes that `B` being stacked on top of `L` is 
%						(part of) the goal state.


% KNOWLEDGE ********************************************************************

% Static knowledge

block(B) :- tic(on(B,_), 0).
location(table).
location(L) :- block(L).

% Describe which actions are executable in a given state
% 	INPUT:		tic(S(...), T)
% 	INPUT:		terminal(T)
% 	OUTPUT: 	executable(A(...), T)

occupied(B, T) :- block(B), tic(on(_,B), T).
free(L, T) :- location(L), time(T), not occupied(L, T).
executable(move(B,L), T) :- block(B), free(B, T), free(L, T), 
							not tic(on(B,L), T), B != L, not terminal(T).

% Describe the effect of state transitions
%	INPUT:		tic(S(...), T)
%	INPUT: 		act(A(...), T)
%	INPUT: 		time(T)
%	OUTPUT:		tic(S(...), T+1)

tic(on(B,L), T+1) :- tic(on(B,L), T), not act(move(B,_), T), time(T).
tic(on(B,L), T+1) :- act(move(B,L),T), time(T).

% Describe the rewards for each state
%	INPUT: 		tic(S(...), T)
%	OUTPUT: 	partialReward(R, T)

tic(goal,T) :- time(T), { not tic(on(B,L), T) : subgoal(B,L) } = 0.    
partialReward(100,T) :- tic(goal,T), not tic(goal,T-1).
partialReward(-1,T)  :- act(_,T-1).


% Describe terminal states
%	INPUT: 		tic(S(...), T)
%	OUTPUT: 	terminal(T)

terminal(T) :- tic(goal,T).
