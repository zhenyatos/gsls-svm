# gsls-svm
Greedy Sparse Least-Squares (GSLS) SVM and how to use it in regression analysis. This algorithm was invented by *Gavin C. Cawley* and *Nicola L.C. Talbot*.

![example](https://user-images.githubusercontent.com/47058532/115428144-c4cbdf00-a20a-11eb-93d4-5cbf84adf1c4.png)

## Requirements
You don't need anything to use `GSLS_SVM` function from **GSLS-SVM.jl** 

But if you want to test it with **test.jl** you should install:\
[Plots](http://docs.juliaplots.org/latest/)\
[Distributions](https://juliastats.org/Distributions.jl/stable/)

## Launch
- From Julia REPL:
```julia
julia> include("main.jl")
```
- from command line (not recommended):
```
julia main.jl
```

## Description
As you might guess, **GSLS** is a **G**reedy algorithm. Its purpose is to construct **S**parse approximation of the **LS**-SVM solution to the regularized least-squares regression problem. Given training data

![data](https://user-images.githubusercontent.com/47058532/115442841-2b58f900-a21b-11eb-975a-4e98db4d6374.gif)

where

![domains](https://user-images.githubusercontent.com/47058532/115442988-58a5a700-a21b-11eb-9a1e-ce5c774a434d.gif)

LS-SVM with kernel function 

![kernel](https://user-images.githubusercontent.com/47058532/115442470-b7b6ec00-a21a-11eb-84ac-9fad31850567.gif)

determines coefficients 

![coefficients](https://user-images.githubusercontent.com/47058532/115443884-83442f80-a21c-11eb-8e48-f53a42c9283c.gif)

for the solution to the mentioned regression problem

![solution](https://user-images.githubusercontent.com/47058532/115445143-23e71f00-a21e-11eb-972a-00c8142eb772.gif)

which minimises LS-SVM objective function. 

We aim to find such an approximation (which we call *sparse*) that for some proper subset (which we call *dictionary*)

![subset](https://user-images.githubusercontent.com/47058532/115445901-2138f980-a21f-11eb-90eb-b5cb41f3360b.gif)

coefficients

![coefficients](https://user-images.githubusercontent.com/47058532/115446537-061ab980-a220-11eb-90bb-cad5bd14eb6d.gif)

of the function

![function](https://user-images.githubusercontent.com/47058532/115446101-62c9a480-a21f-11eb-8099-70da04328ae6.gif)

will minimise GSLS SVM objective function

![objective function](https://user-images.githubusercontent.com/47058532/115446737-4ed27280-a220-11eb-9b4d-352943cd1196.gif)

as much as possible. **γ** is the regularization parameter. At each iteration GSLS chooses some new vector from dataset as support vector, calculates value of the objective function and in a greedy maner, incorporates best possible support vector (on current iteration) to the dictionary, than proceeds to the next iteration. This process is terminated once dictionary has reached some pre-determined size. More detailed description of this simple, but efficient algorithm can be found in [paper](https://www.researchgate.net/publication/221078993_A_Greedy_Training_Algorithm_for_Sparse_Least-Squares_Support_Vector_Machines).

