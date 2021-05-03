on(b1,b2).
on(b2,b3).
on(b3,b4).
on(b4,b5).
on(b5,table).

subgoal(b1,b2).
subgoal(b2,b3).
subgoal(b3,b4).
subgoal(b4,b5).


%tower(T) :- on(T, table).
%towerGreaterEqual(T, T, 1) :- tower(T).
%towerGreaterEqual(T, B, N+1) :- towerGreaterEqual(T, A, N), on(B,A).

%towerHeight(T,N) :- N = #max { X : towerGreaterEqual(T, _, X) }, on(T,table).

%maxTowerHeight(N) :- N = #max { X : towerHeight(_, X) }.

-correct(A) :- on(A,B), subgoal(A,C), B!=C.
correctTowerRec(T, T, 1) :- on(T, table), not -correct(T).
correctTowerRec(T, A, N+1) :- correctTowerRec(T, B, N), on(A,B), subgoal(A,B).
correctTowerHeight(T, N) :- N = #max { 0; X : correctTowerRec(T, _, X) }, on(T,table).
maxCorrectTowerHeight(N) :- N = #max { X : correctTowerHeight(_, X) }.

-clear(A) :- on(B,A), not A=table.
clear(A) :- on(A,_), not -clear(A).
