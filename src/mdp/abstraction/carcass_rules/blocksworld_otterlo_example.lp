%
% 3-blocks world example carcass from Otterlo's PhD thesis (2008) p. 253, example 5.2.1
%
% rule(R,N)										... rule R is the Nth rule to choose from
% applicable(R, (arg1, arg2, ...)) 				... rule $ is applicable with args
% choose(R) 									... rule R was chosen
% abstractAction(abstract(...), ground(...))	... abstract action `abstract(...)` is available and has a corresponding ground action `ground(...)`.

1 { choose(R) : applicable(R, _) } 1.
:- choose(R), rule(R,N), rule(R2,N2), applicable(R2, _), N2 < N.

rule(r1,1).
applicable(r1, (A,B,C)) :- on(A,B), on(B,table), on(C,table), A!=B, B!=C.
abstractAction(move(x1,x3), move(A,C)) :- choose(r1), applicable(r1, (A,B,C)).
abstractAction(move(x3,x1), move(C,A)) :- choose(r1), applicable(r1, (A,B,C)).
abstractAction(move(x1,table), move(A,table)) :- choose(r1), applicable(r1, (A,B,C)).

rule(r2,2).
applicable(r2, (A,B,C)) :- on(A,table), on(B,table), on(C,table), A!=B, B!=C, A!=C.
abstractAction(move(x1,x2), move(A,B)) :- choose(r2), applicable(r2, (A,B,C)).
abstractAction(move(x2,x1), move(B,A)) :- choose(r2), applicable(r2, (A,B,C)).
abstractAction(move(x1,x3), move(A,C)) :- choose(r2), applicable(r2, (A,B,C)).
abstractAction(move(x3,x1), move(C,A)) :- choose(r2), applicable(r2, (A,B,C)).
abstractAction(move(x2,x3), move(B,C)) :- choose(r2), applicable(r2, (A,B,C)).
abstractAction(move(x3,x2), move(C,B)) :- choose(r2), applicable(r2, (A,B,C)).

rule(r3,3).
applicable(r3, (A,B,C)) :- on(A,B), on(B,C), on(C,table), A!=B, B!=C, C!=table.
abstractAction(move(x1,table), move(A,table)) :- choose(r3), applicable(r3, (A,B,C)).

rule(gutter, #sup).	
applicable(gutter, ()).

#show choose/1.
#show abstractAction/2.
