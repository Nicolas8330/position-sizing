"""Position sizing for a small edge: growth, drawdown and the funded-account challenge.

Modules
  rng        the random number generator shared with the JavaScript on the site
  streams    trades and daily P&L turned into day-by-day R (profit / risk to the stop)
  sim_ref    the simulation, one path at a time, written to be read
  simulate   the same simulation for thousands of paths at once (numpy)
  kelly      the Kelly fraction and its bootstrap
  rstats     shape of the R distribution, and clustering of bad trades
  stress     replays of the real worst year and the real worst run of 10 trades
  diversify  correlation of the two rules, and the book that combines them
  analysis   the experiments that feed the page
"""
