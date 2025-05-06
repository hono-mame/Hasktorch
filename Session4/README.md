# Session4
```
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session4-perceptron-andgate"
```

### 4.1 Build and train an AND gate using a simple perceptron

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