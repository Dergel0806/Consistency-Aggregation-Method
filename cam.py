import itertools
import pandas as pd
import numpy as np

data = None # READ EXPERT RESPONSES HERE AS PANDAS DATAFRAME
expImp = None # READ EXPERT IMPORTANCE COEFFICIENTS

def mapToFuzzyNumbers(option):
    '''
    Maps selected option (Strongly agree, Neutral etc.) to fuzzy number via Likert scale.
        
    Arguments:
    `option` - string, should be one of the follows: 'Strongly agree', 'Agree', 'Neutral', 'Disagree' or 'Strongly disagree', otherwise default value of (0.0, 0.05, 0.1) will be applied. 

    Returns: converted fuzzy number, tuple of three floats.
    '''
    if option == 'Strongly agree':
        return (0.6, 0.8, 1.0)
    if option == 'Agree':
        return (0.4, 0.6, 0.8)
    if option == 'Neutral':
        return (0.2, 0.4, 0.6)
    if option == 'Disagree':
        return (0.0, 0.2, 0.4)
    if option == 'Strongly disagree':
        return (0.0, 0.0, 0.2)
    return (0.0, 0.05, 0.1)


def interval(l, n):
    '''
    Split interval l into n equal sub-intervals.
    
    Arguments:
    `l` - tuple that contains two numerical values - start and end of the interval;
    `n` - number of intervals.
    
    Returns:
    list of points that correspond to splitted sub-intervals.
    '''
    w = (l[1] - l[0]) / n
    return [round((l[0] + i * w + l[0] + (i + 1) * w),2) / 2 for i in range(n)]
    
N = 5
xRange = interval([0,1], N)


def memb(x, R):
    '''
    Triangular membership function.
    
    Arguments:
    `x` - point at which membership function is to be evaluated;
    `R` - fuzzy number, tuple of three floats.
    
    Return:
    float number that coresponds to membership function value of fuzzy number `R` at point `x`.
    '''
    a, b, c = R
    if x < a or x > c:
        return 0
    if x < b and x >= a:
        return (x - a) / (b - a)
    if x <= c and x >= b:
        return (b + c - b - x) / (c - b) 
    

def similarity(R1, R2):
    '''
    Similarity between two fuzzy numbers.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers for which similarity will be measured.
    
    Returns:
    float number that represents similarity.
    '''
    mR1 = [memb(x, R1) for x in xRange]
    mR2 = [memb(x, R2) for x in xRange]
    return np.sum(np.minimum(mR1, mR2) ** 2) / np.sum(np.maximum(mR1, mR2) ** 2) 


def hamDist(R1, R2):
    '''
    Hammington distance between `R1` and `R2` fuzzy numbers.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers, distance between which will be measured.
    
    Returns:
    float, distance.
    '''
    mR1 = np.array([memb(x, R1) for x in xRange])
    mR2 = np.array([memb(x, R2) for x in xRange])
    return np.abs(mR1 - mR2) @ xRange


def infDist(R1, R2):
    '''
    Infinum distance between `R1` and `R2` fuzzy numbers.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers, distance between which will be measured.
    
    Returns:
    float, distance.
    '''
    
    return np.min(np.abs(np.array(list(itertools.product(R1, R2)))[:,0] 
            - np.array(list(itertools.product(R1, R2)))[:,1]))
    

def dist(R1, R2):
    '''
    Combined distance (average of Hammington and infinum distances) between `R1` and `R2` fuzzy numbers.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers, distance between which will be measured.
    
    Returns:
    float, distance.
    '''
    
    return (hamDist(R1, R2) + infDist(R1, R2)) / 2


def normDist(R1, R2, R):
    '''
    Normalized distance between `R1` and `R2` fuzzy numbers.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers, distance between which will be measured.
    
    Returns:
    float, distance.
    '''
    
    return dist(R1, R2) / max([dist(Ri, Rj) for Ri, Rj in list(itertools.product(R, R))])


def consMes(R1, R2, R, beta=0.5):
    '''
    Consensus measure between `R1` and `R2` fuzzy numbers in `R` space of all fuzzy numbers that coresponds to selected options in questionary.
    
    Arguments:
    `R1`, `R2` - fuzzy numbers, consensus between which will be measured;
    `R` - list of fuzzy numbers (tuples or lists of three float numbers) that coresponds to selected options in questionary;
    `beta` - relative importance degree between the similarity and the distance to the decision-maker. Default: 0.5.
    
    Returns:
    float, distance.
    '''
    
    return beta * similarity(R1, R2) + (1 - beta) * (1 - normDist(R1, R2, R))


def weightedConsDegree(i, expImp, R=None, consMeasures=None):
    '''
    Weighted consistency degree for expert.
    
    Arguments:
    `i` - index of expert for which weighted consistency degree will be calculated.
    `expImp` - list of degree of importance for each expert. 
    
    `R` - list of fuzzy numbers (tuples or lists of three float numbers) that coresponds to selected options in questionary. Default: None.
    `consMeasures` - Cached list of consistency measures for each fuzzy number. Included for speeding the algorithm. Default: None.
    *Either `R` or `consMeasures` should be passed.*
    
    Returns:
    Weigthed consistency degree, float.
    '''
    
    assert R is not None or consMeasures is not None
    
    if consMeasures is None:
        consMeasure = np.array([consMes(R[i], Rj, R) for Rj in R])
    else:
        consMeasure = consMeasures[i]
    return consMeasure @ expImp


def aggWeight(i, expImp, R=None, weightedConsDegrees=None):
    '''
    Aggregation weight for expert.
    
    Arguments:
    `i` - index of expert for which aggregation weight will be calculated.
    `expImp` - list of degree of importance for each expert. 
    `R` - list of fuzzy numbers (tuples or lists of three float numbers) that coresponds to selected options in questionary. Default: None.
    `weightedConsDegrees` - Cached list of weighted consistency measures for each fuzzy number. Included for speeding the algorithm. Default: None.
    *Either `R` or `weightedConsDegrees` should be passed.*
    
    Returns:
    Aggregation weight, float.
    '''
    
    assert R is not None or consMeasures is not None
    
    if weightedConsDegrees is None:
        weightedConsDegrees = [weightedConsDegree(j, expImp, R=R) for j in range(len(R))]
    return weightedConsDegrees[i] / sum(weightedConsDegrees)


def groupOpinion(R, expImp, weightedConsDegrees):
    '''
    Calculates group opinion.
    
    Arguments:
    `R` - list of fuzzy numbers (tuples or lists of three float numbers) that coresponds to selected options in questionary.
    `expImp` - list of degree of importance for each expert. 
    `weightedConsDegrees` - List of weighted consistency measures for each fuzzy number.
    
    Returns:
    Group opinion, tuple of three elements (fuzzy number).
    '''
    
    weights = [aggWeight(i, R, expImp, weightedConsDegrees) for i in range(len(R))]
    return weights @ np.array(R)


def defuziffication(opinion):
    '''
    Defuzzification.
    
    Arguments:
    `opinion` - group opinion that should be defuzzified.
    
    Returns:
    Defuzzified opinion, float.
    '''
    
    return np.mean(opinion)


def delphi(question, data, expImp, xRange=xRange, S=0.75):
    '''
    Implements consistency aggregation method.
    
    Arguments:
    `question` - label of column in `data` dataframe that contains desired question for evaluation;
    `data` - pandas dataframe with obtained responses;
    `expImp` - list of degree of importance for each expert; 
    `xRange` - list of points to use in membership function evaluation;
    `S` - retainment threshold for rank of each item.
    
    Return:
    Returns tuple of three items, where:
    1. first item is obtained defuzzified rank of question;
    2. second item is obtained conensus rate;
    3. third item is verdict ("Retained" / "Discarder") based on `S` threshold value.
    '''
    
    subData = data[question]
    R = list(subData.apply(mapToFuzzyNumbers)) 
    consMeasures = np.array([[consMes(Ri, Rj, R) for Rj in R] for Ri in R])
    weightedConsDegrees = [weightedConsDegree(j, expImp, R, consMeasures) for j in range(len(R))]
    cons = sum(weightedConsDegrees) / len(subData)
    return defuziffication(groupOpinion(R, expImp, weightedConsDegrees)), cons, 'Retained' if cons >= S else 'Discarded'


ranking = pd.DataFrame(columns=['Name', 'Rank', 'Consensus', 'Verdict'])
for col in data.columns: 
    res = delphi(col, data, expImp, S=0.53)
    ranking.loc[len(ranking)] = [col, res[0], res[1], res[2]]
    

ranking = ranking.sort_values(by=['Rank'], ascending=False)
ranking = ranking.set_index('Name')


ranking.to_csv('results.csv')