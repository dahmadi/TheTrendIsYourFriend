# Double Trouble: Pattern Detector


## Objectives

Our goal is to develop and establish an experimental trading algorithm that identifies short and long position entries, via, trend reversal patterns within daily stock data of 'Invesco QQQ Trust Series 1 (QQQ)'. We will be identifying the two patterns commonly referred to as double tops and double bottoms, which once identified will trigger a buy or sell signal for the algorithm. We then intend to train and test our algorithm in various machine-learning models to select and implement the most efficient machine-learning model for our algorithm.  

## Requirements

- Python Dependencies
    -[alpaca-trade-api]
        - To install alpaca api: 'pip install alpaca-py'
        - Create an account at [Alpaca](https://alpaca.markets/)
        - Create a [Alpaca api key](https://docs.alpaca.markets/docs/getting-started-with-trading-api)
        - Add the Alpaca api key to your .env with the format `ALPACA_API_KEY ={ALPACA_API_KEY}` & `ALPACA_SECRET_KEY ={ALPACA_SECRET_KEY}`

## Our Process

### 1.Pattern Identification
Double top/bottom patterns are technical reversal patterns that form after an asset reaches a high/low price two consecutive times with a moderate decline/incline between the two high/lows.

These patterns are not always easy to spot because there needs to be confirmation with break below support.

Key Elements of a Double Top/Bottom:
- Up/down-trend: price should be moving in an up or down direction.
- Valley for double top pattern, or peak for double bottom pattern.
- Neckline break: horizontal line that is created at the respective.
- Break of Neckline: price drop below/above horizontal neckline. Initiate trigger. 

An example of a double top signal would look something like the following:
![Double Top Example](Images/Double_Top_Example.png)


### 2.Polynomial Smoothing

Polynomial fitting involves using a polynomial equation to approximate a relationship between variables. In our stock data scenario, the polynomial smoothing process was implemented to model and understand the relationship between the stock's historical prices and time. When undergoing the polynomial fitting process a polynomial degree must be selected, we ultimately decided that a polynomial degree of 25 worked best for the two years of 5-minute stock data in our algorithm. The process of fitting polynomials to stock data involves adjusting the coefficients within the equation to minimize differences in a stock's price, at any given time, and the predicted values. This will, in theory, help identify trends and cycles within our dataset.


### 3. Locating Local Min/Max Extrems to Establish Entry Targets



![QQQ plot](Images/Stock_Data_Plot.png)



## Inital Model Selection and Testing