# Session4
```
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session4-perceptron-andgate"
```

## 4.1 Build and train an AND gate using a simple perceptron

Learning rate: 0.1
```
----- Training perceptron of AND gate -----
Initial weights: [2.092598,2.6983223], bias: 2.9987245
Epoch 1 total err: 3.0
Epoch 2 total err: 3.0
Epoch 3 total err: 3.0
Epoch 4 total err: 3.0
Epoch 5 total err: 3.0
Epoch 6 total err: 3.0
Epoch 7 total err: 3.0
Epoch 8 total err: 3.0
Epoch 9 total err: 3.0
Epoch 10 total err: 3.0
Epoch 11 total err: 2.0
Epoch 12 total err: 2.0
Epoch 13 total err: 2.0
Epoch 14 total err: 2.0
Epoch 15 total err: 1.0
Epoch 16 total err: 1.0
Epoch 17 total err: 1.0
Epoch 18 total err: 0.0
Final weights: [0.6925976,0.9983226], bias: -1.1012751
----- Training complete. below are the results -----
Input: [1.0,1.0] Predicted: 1.0 True: 1.0
Input: [1.0,0.0] Predicted: 0.0 True: 0.0
Input: [0.0,1.0] Predicted: 0.0 True: 0.0
Input: [0.0,0.0] Predicted: 0.0 True: 0.0
```

## 4.2 understaing the implementation of Multi-Layer Perceptron in hasktorch

### 1. definition of MLP
```
data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }
```
→ Defines the MLP architecture:  
- feature_counts: A list specifying **the number of units** in each layer.

- nonlinearitySpec: Specifies **the non-linear activation function.**  
  ex) Torch.tanh


```
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

```
→ Represents the MLP model:

- layers: A list of linear layers.

- nonlinearity: The activation function from MLPSpec


### 2. implement the MLP
```
mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
```


### 3. implement XOR function
```
tensorXOR :: Tensor -> Tensor
tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
  where
    a = select 1 0 t
    b = select 1 1 t
```


### 4. training process
① initialize the model
```
init <- sample $ MLPSpec { feature_counts = [2, 2, 1], 
```
input layer: two dimentions  
hidden layer: two dimentions  
output lyaer: one dimention

```
nonlinearitySpec = Torch.tanh } 
```
specify the activation function (tanh)

② training roop
```
trained <- foldLoop init numIters $ \state i -> do
  input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
  let (y, y') = (tensorXOR input, squeezeAll $ model state input)
      loss = mseLoss y y'
  when (i `mod` 100 == 0) $ do
    putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
  (newState, _) <- runStep state optimizer loss 1e-1
  return newState
```
---
a. generate a training data
```
input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
```
if the value is greater than 0.5, then set it to 1. Otherwise, 0 and convert to Float  

---

b. compare the predicted value to actual value
```
let (y, y') = (tensorXOR input, squeezeAll $ model state input)
    loss = mseLoss y y'
```
y: actual value → calculated by tensorXOR (function)  
y': predicted value → calculated by model state input

```
tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
```
---
c. renew the model
```
(newState, _) <- runStep state optimizer loss 1e-1
return newState
・
・
・
optimizer = GD
```
Learning rate: 0.1    
This training uses gardient descent method.  
passes the new model parameter to next iteration.

