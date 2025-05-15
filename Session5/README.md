# Session5
```
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session5-admission"
```

## results of admit.hs
① fininshed Building model and implementing evaluation  
![](charts/MLP_Admission_LearningCurve_1.png)
```haskell
Iteration: 100 | Loss: 0.57774615
Iteration: 200 | Loss: 0.34356028
Iteration: 300 | Loss: 0.20681198
Iteration: 400 | Loss: 0.12736735
Iteration: 500 | Loss: 8.14383e-2
Iteration: 600 | Loss: 5.4999232e-2
Iteration: 700 | Loss: 3.9834276e-2
Iteration: 800 | Loss: 3.1161372e-2
Iteration: 900 | Loss: 2.6212836e-2
Iteration: 1000 | Loss: 2.339447e-2

True Positives: 20.0
True Negatives: 0.0
False Positives: 20.0
False Negatives: 0.0

Predictions: [[0.668993],[0.66907954],[0.6688896],[0.66874707],[0.66882074],[0.6706444],[0.6687267],[0.66872394],[0.66911626],[0.668723],[0.66924596],[0.6687355],[0.6687292],[0.6687647],[0.66872895],[0.66880524],[0.6687602],[0.66906905],[0.6687445],[0.66874284],[0.66872525],[0.66874874],[0.668723],[0.6687224],[0.66872215],[0.66872275],[0.6687222],[0.66872215],[0.66872215],[0.6687224],[0.66872287],[0.6687248],[0.66872215],[0.66872215],[0.66872215],[0.6687224],[0.6687231],[0.66872215],[0.6687222],[0.66872245]]
```
needs to be improved.... :(   
→ The value of predictions looks strange. 
They all appear to converge to the same value.  


memo: 
```haskell 
batchSize = 2  
numIters = 1000  
learningRate = 1e-3  


initModel <- sample $ MLPSpec
    { feature_counts = [7, 4, 2, 1],
      nonlinearitySpec = Torch.sigmoid
    }
```

