import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, sub, transpose2D, sumAll)
import Torch.Functional.Internal (meanAll, powScalar)


ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add (mul input slope) intercept

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

calculateNewA ::
     Tensor ->
     Tensor ->
     Tensor
calculateNewA  estY oldA =
    let diff  = estY - ys
        diff2 = (sumAll $ (mul xs diff)) / 15
  in oldA - diff2 * 2e-5

calculateNewB ::
     Tensor ->
     Tensor ->
     Tensor
calculateNewB estY oldB =
    let diff = estY - ys
        total = sumAll diff / 15
  in oldB - total * 2e-5

train :: Int -> (Tensor, Tensor) -> IO (Tensor, Tensor)
train 0 params = return params
train n (a, b) = do
  let estY = linear (a, b) xs
      loss = asValue (cost estY ys) :: Float
  putStrLn $ "Epoch " ++ show (10 - n) ++ ": Loss = " ++ show loss  -- need to change this too if you change the number of epoch
  let newA = calculateNewA estY a
      newB = calculateNewB estY b
      newAvalue = asValue newA :: Float
      newBvalue = asValue newB :: Float
  putStrLn $ "A: " ++ show newAvalue ++ "B: " ++ show newBvalue
  putStrLn "******************"
  train (n - 1) (newA, newB)
  
main :: IO ()
main = do
  let initialA = asTensor ([5.0] :: [Float])
  let initialB = asTensor ([100.0] :: [Float])
  let epoch = 10  -- do not forget to change the number inside the function "train"
  (finalA, finalB) <- train epoch (initialA, initialB)
  let finalY = linear (finalA, finalB) xs
  -- printOutput finalY
  let finalLoss = asValue (cost finalY ys) :: Float
  putStrLn "---------------------------------------"
  putStrLn $ "Epoch: " ++ show epoch
  putStrLn $ "Final cost: " ++ show finalLoss
  putStrLn $ "Final coefficient A: " ++ show (asValue finalA :: [Float])
  putStrLn $ "Final coefficient B: " ++ show (asValue finalB :: [Float])
  putStrLn "---------------------------------------"