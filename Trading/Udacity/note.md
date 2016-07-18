# Machine Learning for Trading

###### - Mr
###### - Mrs

### Introduction

- Data : [Dowload the dataset HERE](https://dl.dropboxusercontent.com/u/299169754/ml4t.zip)
- See alson : [The wiki](http://quantsoftware.gatech.edu/ML4T_Software_Installation#Data) 

### Data Manipulation

1. Reading & ploting
2. Working with multiple stocks
    + panda.date_range(start, end)
    ```Python
    import panda as pd
    pd.date_range('2014-02-06', '2014-02-06')
    # ['2014-02-06', '2014-02-07', '2014-02-06']
    ```
    + panda.DataFrame.ix
    ```Python
    df.ix['yyyy-mm-dd':'yyyy-mm-dd']
    df.ix['yyyy-mm-dd':'yyyy-mm-dd', ['GOO', 'GLD']]
    ```
    + **Normalizing Stoks**
    ```Python
    df / df.ix[0:] 
    ```
3. Numpy    
    > Numpy is build on top of C & Fortrant, this is why is so fast.
    + Cool attributes
    ```Python
    import numpy as np
    m = ndarray()
    m.shape()
    m.dtype
    m.
    ```
    + opperation
    ```Python
    import numpy as mp
    m.rand.randint(0,10, size=(5,4))
    m.sum(axis=0) #, .max(), .min(), .mean() (recall np.argmax())
    #axis=0 over easch column
    m.shift(1)
    ```
    Remider of the `np.random.seed` package.         
    Remider of the `time` package
    + Assignement and slizing and boolean mask
    ```Python
    m[:, 3] = [1, 2, 3, 4, 5]
    m[indices] # with indices numpy.ndarray
    m[m > mean] # with mean Int
    np.dot(m,p)

    ```
4. Statistics
    + Rolling Statistics
    ```Python
    ```
    `pandas.stats.moment`
    + Daily returns
    ```Python
    daily[t] = (price[t] / price[t-1]) -1
     ```

    *Note une image*
### Computational Investing


### Learning Algorithms for trading
