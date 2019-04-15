# outdated

---


## UP2NOW

### done
* **Class** : DataSamples
  * function: **generateWithDeadzone**
    * create 1D data array for half a period
    * return this array
  * function: **extendData**
    * create 1D data array for one full period
    * return this array
* **Class** : DataSamples_Type2 as subclass of DataSamples
  * function: **generate**
    * create another type
    * return this array  
  * function：**extendData**
    * create for one full period
    * return this array
* **IO2csv**:some functions for IOs
* **WithSliding**:expand_dims and squeeze original data scale
  * current:four-fold features

### remains sorted by time asc

* Activation -> 'ReLU' will lead to zeros under this situation: so comes to 'sigmoid'
* over-fitting: **Drop-out**
* modelling it 
* how to label thousands of training data sets
  * label them with 0-1 so the output should be regarded as its probility



outdated

---

***



* feature?

  * online or offline

  * Ia(phase) or whatever

  * > ia

* what is deadtime

  *  features

* how to compensate 
  * ideal output
  * actual situation
  	* uncompensated
  	* overcompensated

* Keras or tensorflow

  >  tensorflow

* 1 or 2 dimensional data structure

> 1dimensional

* TODO
* neural networks
  * CNN-like
    * convolution
    * pooling
    * full connected layer
  * which activation function to choose and the meaning of using activation functions
    * features or costs
    * two examples: ReLU and sigmod:  

