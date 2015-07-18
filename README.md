# Loïc m. Divad notebook

## IPython notebook
The IPython Notebook is an interactive computational environment, in which you can combine code execution, rich text, mathematics, plots and rich media. It aims to be an agile tool for both exploratory computation and data analysis, and provides a platform to support reproducible research, since all inputs and outputs may be stored in a one-to-one way in notebook documents.

![IPython Logo](https://dl.dropboxusercontent.com/s/c9te64xd06a36lw/IPy_header.png?)

## SPARK as a trigger for this.
Since the code can be executed from my personnal laptop or from my server I need to keep up to date my sources code on the both sides. Plus, it's so much easier to commit some files to run to a server than upload them by FTP.
*But, why store scripts on a server in a first place?*
**SPARK** is the answer... Since I don't have hadoop and Spark on my laptop I work "on line". And it's allow you to do some realy cool things: 
```python
@classmethod
def mapByCart(self, prod=False):
        """ Simple MapReduce on rough vehicle arround Paris
            No Return. CSV: ...;type of vehicle;... """
        res = self.rdd.map(lambda line: line.split(";"))\
            .filter(lambda y: y[0] != '')\
            .map(lambda y: (y[10], 1))\
            .reduceByKey(lambda a, b: a + b)\
            .sortBy(lambda y: y[1])
            
        res.saveAsTextFile("hdfs:/" + self.target + folder)
```
## The Labs folder
Most of Lab report implies a simple python script. It cant also be stored here.
See also: [the Labs report repo](https://github.com/DivLoic/TP-LabSession/blob/master/README.md) https://github.com/DivLoic/TP-LabSession

