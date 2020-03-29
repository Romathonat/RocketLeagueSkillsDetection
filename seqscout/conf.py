# this file holds the default value for problem configuration
TOP_K = 5
ITERATIONS_NUMBER = 1000
TIME_BUDGET = 10000
THETA = 0.8
DATA = 'promoters' # promoters, sc2, splice, context, block, skating, jmlr, aslbu
QUALITY_MEASURE = 'WRAcc' # WRAcc, F1, Informedness
NUMERIC_REMOVE_PROBA = 0.9
CROSS_VALIDATION_NUMBER = 5
CLASSIFICATION_ALGORITHM = "RF" # RF, DT, SVM, XGB, NB

# Do not change this value unless you do not mind if xp give wrong values !
TIME_BUDGET_XP = 2**30
BEAM_WIDTH = 50

