% Domain knowledge

goalRelevantBlock(A) :- subgoal(A, _).
goalRelevantBlock(A) :- subgoal(_, A).

above(A,B) :- on(A,B).
above(A,B) :- on(A,C), above(C,B).

towerHeight(T,N+1) :- on(T, table), N = #count { A : above(A,T) }.
maxTowerHeight(N) :- N = #max { X : towerHeight(_, X) }.

goodPartialTowerBase(TowerBase) :- subgoal(_, TowerBase),
								   not subgoal(TowerBase, _),
				   	   			   0 = #count { A : above(TowerBase, A), goalRelevantBlock(A) }.

goodPartialTower(TowerBase, TowerBase) :-  goodPartialTowerBase(TowerBase).
goodPartialTower(TowerBase, B) :- goodPartialTower(TowerBase, B2), on(B,B2), subgoal(B,B2).
goodPartialTowerTop(TowerBase, TopBlock) :- goodPartialTower(TowerBase, TopBlock), 
											subgoal(B, TopBlock),
											not on(B,TopBlock).

clear(A) :- on(A,_), not on(_,A).
clear(A) :- table = A.

% Carcass rules
%
% rule(R,N)										... rule R is the Nth rule to choose from
% applicable(R, (arg1, arg2, ...)) 				... rule $ is applicable with args
% choose(R) 									... rule R was chosen
% abstractAction(abstract(...), ground(...))	... abstract action `abstract(...)` is available and has a corresponding ground action `ground(...)`.

1 { choose(R) : applicable(R, _) } 1.
:- choose(R), rule(R,N), rule(R2,N2), applicable(R2, _), N2 < N.


% Rule 0: a goal-tower exists and is clear, next goal block is clear
rule(r0,0).
applicable(r0, (GoodTowerTop, NextGoalBlock)) :- goodPartialTowerTop(_, GoodTowerTop),
												  clear(GoodTowerTop),
												  subgoal(NextGoalBlock, GoodTowerTop),
												  clear(NextGoalBlock).
abstractAction(move(nextGoalBlock, goodTowerTop), move(NextGoalBlock, GoodTowerTop)) :- choose(r0),
																						applicable(r0, (GoodTowerTop, NextGoalBlock)).
												 

% Rule 1: A goal tower exists and has other blocks on top. 
rule(r1, 1).
applicable(r1, (GoodTowerBase, GoodTowerTop, BadTopBlock)) :- goodPartialTowerTop(GoodTowerBase, GoodTowerTop),
												 			  not clear(GoodTowerTop),
												              above(BadTopBlock, GoodTowerTop),
												              clear(BadTopBlock).
abstractAction(move(badTopBlock, table), move(BadTopBlock, table)) :- choose(r1), applicable(r1, (_, _, BadTopBlock)).
abstractAction(move(badTopBlock, otherTowerTop), move(BadTopBlock, OtherTowerTop)) :- choose(r1), 
																					  applicable(r1, (GoodTowerBase, GoodTowerTop, BadTopBlock)),
																					  clear(OtherTowerTop),
																					  OtherTowerTop != BadTopBlock,
																					  OtherTowerTop != table.



% Rule 2:     a goal-tower exists and is clear, next goal block is not clear.
rule(r2,2).
applicable(r2, (GoodTowerTop, BadTopBlock)) :- 	goodPartialTowerTop(GoodTowerBase, GoodTowerTop),
											    clear(GoodTowerTop),
												subgoal(NextGoalBlock, GoodTowerTop),
												not clear(NextGoalBlock),
												above(BadTopBlock,NextGoalBlock),
												clear(BadTopBlock).
abstractAction(move(badTopBlock, table), move(BadTopBlock, table)) :- choose(r2), applicable(r2, (_, BadTopBlock)).
abstractAction(move(badTopBlock, otherTowerTop), move(BadTopBlock, OtherTowerTop)) :- choose(r2), 
																					  applicable(r2, (GoodTowerTop, BadTopBlock)),
																					  clear(OtherTowerTop),
																					  OtherTowerTop != GoodTowerTop,
																					  OtherTowerTop != BadTopBlock,
																					  OtherTowerTop != table.

% Rule 3:	No gloal tower exists, first goal block is not clear.
rule(r3,3).
applicable(r3, (BadTopBlock)) :- not goodPartialTowerBase(_),
								 subgoal(_, TowerBase),
								 not subgoal(TowerBase, _),
								 not clear(TowerBase),
								 above(BadTopBlock, TowerBase),
								 clear(BadTopBlock).
abstractAction(move(badTopBlock, table), move(BadTopBlock, table)) :- choose(r3), applicable(r3, (BadTopBlock)).
abstractAction(move(badTopBlock, otherTowerTop), move(BadTopBlock, OtherTowerTop)) :- choose(r3), 
																	 				  applicable(r3, (BadTopBlock)),
																	 				  clear(OtherTowerTop),
																	 				  OtherTowerTop != BadTopBlock,
																	 				  OtherTowerTop != table.


% Rule 4:	No goal tower exists, first goal block is clear.
rule(r4,4).
applicable(r4, (TowerBase)) :- not goodPartialTowerBase(_),
							   subgoal(_, TowerBase),
							   not subgoal(TowerBase, _),
							   clear(TowerBase).

abstractAction(move(towerBase, table), move(TowerBase, table)) :- choose(r4), applicable(r4, (TowerBase)).

% The gutter action is available in every state and collects all available actions that were not covered by the rule-specific abstract actions.
abstractAction(gutterAction, move(X, Y)) :- clear(X), clear(Y), X!=Y, X!=table, not abstractAction(move(_,_), move(X, Y)), not on(X,Y).

% The gutter state collects all states not covered by some rule.
rule(gutterState, #sup).	
applicable(gutterState, ()).

#show choose/1.
#show abstractAction/2.
%#show applicable/2.
%#show clear/1.
%#show goodPartialTowerTop/2.
%#show goodPartialTowerBase/1.
%#show goodPartialTower/2.
