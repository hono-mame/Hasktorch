import Torch.Tensor (Tensor, asTensor, asValue, numel)
import Torch.Functional (matmul, mul, add, sub, transpose2D, sumAll)
import Torch.Functional.Internal (meanAll, powScalar)
import Control.Monad (when)

ys :: Tensor -- 売上高
ys = asTensor ([123, 290, 230, 261, 140, 173, 133, 179, 210, 181] :: [Float])
x1s :: Tensor -- 乗降客数
x1s = asTensor ([93, 230, 250, 260, 119, 183, 151, 192, 263, 185] :: [Float])
x2s :: Tensor -- 取扱品目数
x2s = asTensor ([150, 311, 182, 245, 152, 162, 99, 184, 115, 105] :: [Float])

epoch :: Int
epoch = 150
learningRate :: Tensor
learningRate = 2e-5


linear :: 
    (Tensor, Tensor, Tensor) -> -- (a1, a2, b)
    (Tensor, Tensor) ->         -- (x1s, x2s)
    Tensor                      -- z
linear (a1, a2, b) (x1s, x2s) = add (add (mul x1s a1) (mul x2s a2)) b

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = 
    let diffs = sub z z'
        squarediffs = mul diffs diffs
        squarediffsList = asValue squarediffs :: [Float]
        answer = (sum squarediffsList) / fromIntegral (length squarediffsList)
        answerT = asTensor answer
  in answerT

printOutput :: 
    Tensor ->
    IO ()
printOutput estimatedY =
    let ysList = asValue ys :: [Float]
        estimatedYList = asValue estimatedY :: [Float]
  in mapM_ (\(y, e) -> 
      putStrLn $ "correct answer: " ++ 
      show y ++ "\nestimated: " ++ 
      show e ++ "\n*******")
      (zip ysList estimatedYList)

calculateNewA1 ::
    Tensor -> 
    Tensor ->
    Tensor
calculateNewA1 estY oldA1 =
    let diff = estY - ys
        n = asTensor (fromIntegral (numel ys) :: Int) :: Tensor
        grad = sumAll (mul x1s diff) / n
    in oldA1 - grad * learningRate


calculateNewA2 ::
    Tensor -> 
    Tensor -> 
    Tensor
calculateNewA2 estY oldA2 =
    let diff = estY - ys
        n = asTensor (fromIntegral (numel ys) :: Int) :: Tensor
        grad = sumAll (mul x2s diff) / n
    in oldA2 - grad * learningRate

calculateNewB ::
    Tensor ->
    Tensor -> 
    Tensor
calculateNewB  estY oldB =
    let diff = estY - ys
        n = asTensor (fromIntegral (numel ys) :: Int) :: Tensor
        grad = sumAll diff / n
    in oldB - grad * learningRate

train ::
    Int -> -- epoch
    (Tensor, Tensor, Tensor) -> -- (a1, a2, b)
    [Tensor] -> -- losses
    IO ((Tensor, Tensor, Tensor), [Tensor])
train  0 params losses = return (params, reverse losses)
train  n (a1, a2, b) losses = do
  let estY = linear (a1, a2, b) (x1s, x2s)
      currLoss = cost estY ys
  when ((epoch - n) `mod` 10 == 0) $
    putStrLn $ "Epoch " ++ show (epoch - n) ++ ": Loss = " ++ show (asValue currLoss :: Float)
  let newA1 = calculateNewA1 estY a1
      newA2 = calculateNewA2 estY a2
      newB  = calculateNewB  estY b
  train  (n - 1) (newA1, newA2, newB) (currLoss : losses)
  
main :: IO ()
main = do
  let initialA1 = asTensor ([0.0] :: [Float])
  let initialA2 = asTensor ([0.0] :: [Float]) 
  let initialB = asTensor ([0.0] :: [Float])
  ((finalA1, finalA2, finalB), losses) <- train epoch (initialA1, initialA2, initialB) []
  let finalY = linear (finalA1, finalA2, finalB) (x1s, x2s)
  -- printOutput finalY ys
  let finalLoss = asValue (cost finalY ys) :: Float
  putStrLn "---------------------------------------"
  putStrLn $ "Epoch: " ++ show epoch
  putStrLn $ "Final cost: " ++ show (asValue (last losses) :: Float)
  putStrLn $ "Final coefficient A1: " ++ show (asValue finalA1 :: [Float])
  putStrLn $ "Final coefficient A2: " ++ show (asValue finalA2 :: [Float])
  putStrLn $ "Final coefficient B: " ++ show (asValue finalB :: [Float])
  putStrLn "---------------------------------------"