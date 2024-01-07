# Double Trouble: Pattern Detector

"""
Psuedo:
- read in data using alpacas (5 min chart). starting with QQQ
- format data for usage. variables to use OHLCV (open high low close volume?)
- code to establish what a double top or double bottom looks like (later on we can include other patterns)
- machine learning neural network to teach identification of the pattern.
- each compose independent neural network to identify what parameters work best.
- once model has learned to identify patterns implement a buy.
- program a trend (for upward trend we're working with higher highs (HH) and higher lows(HL) for downward trend we're working with lower highs (LH) and lower lows(LL))... like a staircase
- implement algorithm on QQQ
- report on profitability
- comapare and contrast to buy and hold trading method. and any other methods?
- what are the next steps? do we compare to another algo? another portfolio? compare to BRK?
"""

## Objectives
Our goal is to develop and establish an expiremental trading algorithm that identifies short and long position entries, via, trend reversal patterns within daily stock data of 'Invesco QQQ Trust Series 1 (QQQ)'. We will be identifying the two pattern's commonly referred to as double tops and double bottoms, which once identified will trigger a buy or sell signal for the algorithm. We then indend to train and test our alrorithm in various machine learning models with the intention to select and implement the most efficient machine learning model for our algorithm. 

## Requirements



## Pattern Identification
Double top/bottom patterns are technical reversal patterns that form after an asset reaches a high/low price two consecutive times with a moderate decline/incline between the two high/lows.

These patterns are not always easy to spot because there needs to be confirmation with break below support.

Key Elements of a Double Top/Bottom:
- Up/down-trend: price should clearly be moving in an up or down direction.
- Valley for double top pattern, or peak for double bottom pattern.
- Neckline break: horizontal line that is created at the respective.
- Break of Neckline: price drop below/above horizontal neckline. Initiate trigger. 
![Double Top Example](Images/Double_Top_Example.png)
