# Connect4
AI project Connect4 with Minimax and Alpha Beta Pruning.
CS383 Homework Project

In my evaluation function, for every rows, columns and diagonals, I checked if
there are 2 or more consecutive chessman and give them 10 to a power of the number
of consecutive chessman or deduct in other case.
For example -if there are no consecutive chessman - 10 ** 1
if there are two consecutive chessman - 10 ** 2
if there are three consecutive chessman - 10 ** 3 and more on.

And we check the evaluation number line by line of the row, Col, and diagonal.
For example , if one of the list contains the state of [ 1,0,-1,1,1], the value of the line will
be = 100

I evaluate the value for columns, rows and diagonals and assign it and combine them.
There are only simple “my_test” in test_board.py but my code satisfies every
state including if I want to check from the start state.
