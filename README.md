# Loïc m. Divad notebook

## Jupyter notebook
The IPython Notebook is an interactive computational environment, in which you can combine code execution, rich text, mathematics, plots and rich media. It aims to be an agile tool for both exploratory computation and data analysis, and provides a platform to support reproducible research, since all inputs and outputs may be stored in a one-to-one way in notebook documents.

![IPython Logo](https://dl.dropboxusercontent.com/s/3a831t8txwp2nxl/jup_header.png?dl=0)

## R    
Thanks to [IRKernel](http://irkernel.github.io), Jupyter kernel for the R programming language the notbook project contains also R code. Here is the intallation cmd line:   
```R
install.packages(c('repr', 'pbdZMQ', 'devtools')) # repr is already on CRAN
devtools::install_github('IRkernel/IRdisplay')
devtools::install_github('IRkernel/IRkernel')
IRkernel::installspec()
```
*to be continued...*

## Scala
Since Scala is a great family language with an interactive tool (REPL), working with jupyter is a awesome experience. To make it happen i chose [jupyter-scala](https://github.com/alexarchambault/jupyter-scala).

```bash
$ cd jupyter-scala
$ pip install --upgrade "ipython[all]"
$ ipython kernelspec list
$ jupyter-scala
```
Here is a simple jupyter cell with scala code.
```{scala}
import scala.math._
println sys.env("HOME")
val t = 0.1
val foo = for(i <- 0 until 10 by 2) yield {
  pow(t, i)*exp(-t)
}
```

## Spark
*...*

## The Labs folder
Most of Lab report implies a simple python script. It cant also be stored here.
See also: [the Labs report repo](https://github.com/DivLoic/TP-LabSession/blob/master/README.md) https://github.com/DivLoic/TP-LabSession

## Dato folder
*...*
