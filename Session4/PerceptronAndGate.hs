import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, sub)
import Control.Monad (foldM)
import System.Random

---------- initial values ----------
epoch :: Int
epoch = 100
learningRate :: Float
learningRate = 0.1
------------------------------------

trainingData :: [([Float], Float)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]


step :: -- activation function
    Tensor ->
    Tensor
step net =
    if (asValue net :: Float) > 0 
        then asTensor(1 :: Float)
    else asTensor(0 :: Float)


perceptron ::
    Tensor -> -- x
    Tensor -> -- weights
    Tensor -> -- bias
    Tensor    -- output
perceptron x weight bias = add (matmul x weight) bias


calculateError ::
  Tensor -> -- true
  Tensor -> -- predicted
  Tensor -- error
calculateError y_true y_pred = sub y_true y_pred


trainEachSample :: 
    Tensor ->  -- weights
    Tensor ->  -- bias
    ([Float], Float) -> -- input and label  
    IO (Tensor, Tensor, Float) -- updated weights, bias and error
trainEachSample weight bias (inputs, label) = do
    let x = asTensor inputs
        y_true = asTensor label
        y_pred = step $ perceptron x weight bias
        err = calculateError y_true y_pred
        errorValue = abs (asValue err :: Float)
        newWeight = add weight (mul (mul (asTensor learningRate) x) err)
        newBias = add bias (mul (asTensor learningRate) err)
    return (newWeight, newBias, errorValue)


-- Train on all samples for a single epoch
trainEpoch :: 
    Tensor -> -- weights
    Tensor -> -- bias
    IO (Tensor, Tensor, Float) -- updated weights, bias and error
trainEpoch weight bias =
    foldM update (weight, bias, 0.0) trainingData
  where
    update (w, b, totalErr) sample = do
        (newWeight, newBias, err) <- trainEachSample w b sample
        return (newWeight, newBias, totalErr + err)


printEpoch :: 
    Int -> -- the number of current epoch
    Float -> -- total error
    IO () 
printEpoch epochNum totalError = 
    putStrLn $ "Epoch " ++ show epochNum ++ " total err: " ++ show totalError


train :: 
    Tensor -> -- weights
    Tensor -> -- bias
    Int -> -- number of epochs
    IO (Tensor, Tensor) -- updated weights and bias
train weight bias 0 = return (weight, bias)
train weight bias n = do
    (newWeight, newBias, newTotalError) <- trainEpoch weight bias  
    let currentEpoch = epoch - n + 1
    printEpoch currentEpoch newTotalError
    if newTotalError == 0
        then return (newWeight, newBias)
        else train newWeight newBias (n - 1)


main :: IO ()
main = do
    -- initialize parameters(weights and bias) with random values.
    w1 <- randomRIO (-3.0, 3.0) :: IO Float
    w2 <- randomRIO (-3.0, 3.0) :: IO Float
    b <- randomRIO (-3.0, 3.0) :: IO Float
    
    let weight = asTensor [w1, w2]
        bias = asTensor b
    
    putStrLn("----- Training perceptron of AND gate -----")
    putStrLn $ "Initial weights: " ++ show [w1, w2] ++ ", bias: " ++ show b
    (trainedWeight, trainedBias) <- train weight bias epoch
    
    let finalWeight = asValue $ trainedWeight
        finalBias = asValue trainedBias
    putStrLn $ "Final weights: " ++ show (finalWeight :: [Float]) ++ ", bias: " ++ show (finalBias :: Float)
    
    putStrLn "----- Training complete. below are the results -----"
    mapM_
        (\(inputs, label) -> do
            let x = asTensor inputs
                net = perceptron x trainedWeight trainedBias
                y_pred = step net
            putStrLn $ "Input: " ++ show inputs ++
                       " Predicted: " ++ show (asValue y_pred :: Float) ++
                       " True: " ++ show label)
        trainingData


