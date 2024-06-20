teaching an artificial neural network to play tic tac toe

uses layer and network modules from https://github.com/beauxq/learn-ann
(`$ pip install -r requirements.txt`)

---

At one point, after lots of training, the neural network found themself as `X` in this situation:

    _ _ X
    O X O
    _ _ _

They could have taken the bottom left and immediately won, but they thought both the top left and top middle were better, with the top middle being the better of those two.

    _ X X
    O X O
    _ _ _

This is almost understandable, because this move makes 3 opportunities to win. But what makes this confusing is that the move chosen is not a guaranteed win.
If `X` played very poorly, and `O` played perfectly, `O` can still win.

    _ X X    _ X X    O X X
    O X O    O X O    O X O
    O _ _    O _ X    O _ X

The other option that `X` most considered did guarantee a win, even if `O` played perfectly and `X` played as poorly as possible.

    X _ X    X _ X    X _ X    X O X    X O X
    O X O    O X O    O X O    O X O    O X O
    _ _ _    O _ _    O X _    O X _    O X X

Even if `X` learned to anticipate their future moves, they should have learned that there was a random number generator involved which was meant to keep them from playing perfectly.

Maybe `X` learned to decipher the random number generator, and thus knew that they were guaranteed to win...
