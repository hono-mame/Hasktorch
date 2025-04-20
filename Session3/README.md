# Hands On Tasks

How to execute the LinearRegression projects:
```
(if necessary)
$ docker-compose up -d

$ docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session3-linear-regression"
```

### 3.b: results of initial linear function
correct answer: 130.0  
estimated: 176.72504  
******* 
correct answer: 195.0  
estimated: 197.81503  
*******
correct answer: 218.0  
estimated: 249.43002
*******
correct answer: 166.0  
estimated: 193.93002  
*******
correct answer: 163.0  
estimated: 214.46503
*******
correct answer: 155.0  
estimated: 165.07004
*******
correct answer: 204.0  
estimated: 178.94504
*******
correct answer: 270.0  
estimated: 203.36502
*******
correct answer: 205.0  
estimated: 164.51503
*******
correct answer: 127.0  
estimated: 137.87503
*******
correct answer: 260.0  
estimated: 211.69003
*******
correct answer: 249.0  
estimated: 238.33002
*******
correct answer: 251.0  
estimated: 236.11005
*******
correct answer: 158.0  
estimated: 158.41003
*******
correct answer: 167.0  
estimated: 190.60004
*******

```
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, transpose2D)


ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add (mul input slope) intercept

main :: IO ()
main = do
  let sampleA = asTensor ([0.555] :: [Float])
  let sampleB = asTensor ([94.585026] :: [Float])

  let estimatedY = linear (sampleA, sampleB) xs

  let ysList = asValue ys :: [Float]  -- convert Tensor to [Float]
  let estimatedYList = asValue estimatedY :: [Float]  -- convert Tensor to [Float]

  -- output
  mapM_ (\(y, e) -> 
    putStrLn $ "correct answer: " ++ 
    show y ++ "\nestimated: " ++ 
    show e ++ "\n*******")
    (zip ysList estimatedYList)
```