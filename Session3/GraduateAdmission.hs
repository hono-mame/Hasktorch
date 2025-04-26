{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import GHC.Generics (Generic)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv
import Torch.Tensor (Tensor, asTensor, asValue, numel)
import Torch.Functional (matmul, mul, add, sub, transpose2D, sumAll, exp)
import Torch.Functional.Internal (meanAll, powScalar)
import ML.Exp.Chart (drawLearningCurve)

-- structure of input CSV
data Input = Row
  { x :: !Float   -- row 1
  , y :: !Float   -- row 2
  } deriving (Show, Eq, Generic)

instance FromRecord Input

-- read CSV file and convert to Tensors
loadXY :: FilePath -> IO (Tensor, Tensor)
loadXY filePath = do
  csvData <- BL.readFile filePath
  case decode HasHeader csvData of  -- ヘッダーありCSVを読み込む
    Right v -> do
      let xsList = V.toList $ V.map x v  -- get x as a list
          ysList = V.toList $ V.map y v  -- get y as a list
          xs = asTensor xsList           
          ys = asTensor ysList           
      return (xs, ys)

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add (mul input slope) intercept



sigmoid :: (Tensor, Tensor) -> Tensor -> Tensor
sigmoid (slope, intercept) input =
  let z = input * slope + intercept
  in 1 / (1 + Torch.Functional.exp (-z))


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
    Tensor ->
    IO ()
printOutput estimatedY ys =
    let ysList = asValue ys :: [Float]
        estimatedYList = asValue estimatedY :: [Float]
  in mapM_ (\(y, e) -> 
      putStrLn $ "correct answer: " ++ 
      show y ++ "\nestimated: " ++ 
      show e ++ "\n*******")
      (zip ysList estimatedYList)

calculateNewA ::
     Tensor -> -- xs
     Tensor -> -- ys
     Tensor -> -- estY
     Tensor -> -- oldA
     Tensor    -- newA
calculateNewA xs ys estY oldA =
    let diff  = estY - ys
        n = asTensor (fromIntegral (numel ys) :: Int) :: Tensor
        diff2 = (sumAll $ (mul xs diff)) / 320
  in oldA - diff2 * 2e-5

calculateNewB ::
     Tensor -> -- xs
     Tensor -> -- ys
     Tensor -> -- estY
     Tensor -> -- oldB
     Tensor    -- newB
calculateNewB xs ys estY oldB =
    let diff = estY - ys
        n = asTensor (fromIntegral (numel ys) :: Int) :: Tensor
        total = sumAll diff / n
  in oldB - total * 2e-5

  

train ::
    Tensor -> -- xs
    Tensor -> -- ys
    Int ->    -- epoch
    (Tensor, Tensor) -> -- params (a, b)
    [Tensor] ->         -- accumulated losses
    IO ((Tensor, Tensor), [Tensor])
train xs ys 0 params losses = return (params, reverse losses)
train xs ys n (a, b) losses = do
  let estY = sigmoid (a, b) xs
      currLoss = cost estY ys
      lossVal = asValue currLoss :: Float
  putStrLn $ "Epoch " ++ show (300 - n) ++ ": Loss = " ++ show lossVal
  let newA = calculateNewA xs ys estY a
      newB = calculateNewB xs ys estY b
      newAvalue = asValue newA :: Float
      newBvalue = asValue newB :: Float
  putStrLn $ "A: " ++ show newAvalue ++ " B: " ++ show newBvalue
  putStrLn "******************"
  train xs ys (n - 1) (newA, newB) (currLoss : losses)


main :: IO ()
main = do
  (xs, ys) <- loadXY "Session3/data/train.csv"
  -- putStrLn "xs:"
  -- print xs
  -- putStrLn "ys:"
  -- print ys
  let initialA = asTensor ([0.0] :: [Float])
  let initialB = asTensor ([0.0] :: [Float])
  let epoch = 300  -- do not forget to change the number inside the function "train"
  ((finalA, finalB), losses) <- train xs ys epoch (initialA, initialB) []
  let finalY = sigmoid (finalA, finalB) xs
  -- printOutput finalY ys
  let finalLoss = asValue (cost finalY ys) :: Float
  putStrLn "---------------------------------------"
  putStrLn $ "Epoch: " ++ show epoch
  putStrLn $ "Final cost: " ++ show (asValue (last losses) :: Float)
  putStrLn $ "Final coefficient A: " ++ show (asValue finalA :: [Float])
  putStrLn $ "Final coefficient B: " ++ show (asValue finalB :: [Float])
  putStrLn "---------------------------------------"

  (xEval, yEval) <- loadXY "Session3/data/eval.csv"
  let predEvalY = sigmoid  (finalA, finalB) xEval
  printOutput predEvalY yEval

  -- Learning Curve 
  -- let lossValues = map (\t -> asValue t :: Float) losses
  -- putStrLn "Loss values:"
  -- print lossValues
  -- needs to be fixed (does not work)
  -- drawLearningCurve "app/Session3/charts/GraduateAdmissionSigmoidLearningCurve.png" "Learning Curve" [("Training Loss", lossValues)]
