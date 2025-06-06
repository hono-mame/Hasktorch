{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv (decodeByName, FromNamedRecord)
import ML.Exp.Chart (drawLearningCurve)
import Evaluation (evalAccuracy, evalPrecision, evalRecall, calcF1, evalMacroF1, evalWeightedF1, evalMicroF1, confusionMatrix, confusionMatrixPairs)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (mul, add, sub, sigmoid, logSoftmax, nllLoss', binaryCrossEntropyLoss')
import Torch.Device (Device)
import Control.Monad (forM, forM_)

data Input = Input
  { greScore :: Float,
    toeflScore :: Float,
    universityRating :: Float,
    sop :: Float,
    lor :: Float,
    cgpa :: Float,
    research :: Float,
    chanceOfAdmit :: Float
  } deriving (Show, Generic, FromNamedRecord)


data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  } deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) = scanl shift (a, b) t
      mkLayerSizes _ = []
      shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x


loadAdmissionData :: FilePath -> IO (Tensor, Tensor)
loadAdmissionData filePath = do
  csvData <- BL.readFile filePath
  case decodeByName csvData of
    Left err -> error $ "Failed to decode CSV: " ++ err
    Right (_, v) -> do
      let inputs = V.toList $ V.map (\input -> 
                    [ greScore input
                    , toeflScore input
                    , universityRating input
                    , sop input
                    , lor input
                    , cgpa input
                    , research input
                    ]) v
          targets = V.toList $ V.map (\input -> [chanceOfAdmit input]) v
          
          inputTensor = asTensor inputs
          targetTensor = asTensor targets
      
      return (inputTensor, targetTensor)

batchSize = 2
numIters = 400
learningRate = 1e-3

trainMLP :: MLP -> Tensor -> Tensor -> IO MLP
trainMLP initModel inputs targets = do
  (trainedModel, lossValues) <- foldLoop (initModel, []) numIters $ \(state, losses) i -> do
    let yPred = mlp state inputs
        loss = mseLoss targets yPred
        lossValue = asValue loss :: Float
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show lossValue
    (newState, _) <- runStep state optimizer loss learningRate
    return (newState, losses ++ [lossValue])
  
  drawLearningCurve "Session5/charts/MLP_Admission_LearningCurve.png" "Learning Curve" [("Training Loss", lossValues)]
  return trainedModel
  where
    optimizer = GD


trainMLP_crossEntropy :: MLP -> Tensor -> Tensor -> IO MLP
trainMLP_crossEntropy initModel inputs targets = do
  (trainedModel, lossValues) <- foldLoop (initModel, []) numIters $ \(state, losses) i -> do
    let yPred = sigmoid (mlp state inputs)
        loss = binaryCrossEntropyLoss' yPred targets
        lossValue = asValue loss :: Float
    when (i `mod` 300 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show lossValue
    (newState, _) <- runStep state optimizer loss learningRate
    return (newState, losses ++ [lossValue])
  
  drawLearningCurve "Session5/charts/MLP_Admission_LearningCurve_CrossEntropy.png" "Learning Curve (Binary Cross Entropy)" [("Training Loss", lossValues)]
  return trainedModel
  where
    optimizer = GD

evaluateModel :: MLP -> Tensor -> Tensor -> IO ()
evaluateModel trainedModel inputs targets = do
  let predictions = sigmoid (mlp trainedModel inputs)
      predictedLabels = toType Float (gt predictions 0.675)
      trueLabels = toType Float (gt targets 0.675)
  putStrLn $ "targets: " ++ show (asValue targets :: [[Float]])

  let tp = sumAll (mul predictedLabels trueLabels)
      tn = sumAll (mul (1 - predictedLabels) (1 - trueLabels))
      fp = sumAll (mul predictedLabels (1 - trueLabels))
      fn = sumAll (mul (1 - predictedLabels) trueLabels)

  let cm = confusionMatrix tp tn fp fn 
      cmPairs = confusionMatrixPairs cm

  putStrLn $ "Predictions: " ++ show (asValue predictions :: [[Float]])
  putStrLn $ "Predicted Labels: " ++ show (asValue predictedLabels :: [[Float]])
  putStrLn $ "True Labels: " ++ show (asValue trueLabels :: [[Float]])

  mapM_ (\(i, (tp', tn', fp', fn')) -> do
            let acc = evalAccuracy tp' tn' fp' fn'
                prec = evalPrecision tp' fp'
                recall = evalRecall tp' fn'
                f1 = calcF1 prec recall
            putStrLn $ "\nPair " ++ show i ++ ":"
            putStrLn $ "  True Positive: " ++ show (asValue tp' :: Float)
            putStrLn $ "  True Negative: " ++ show (asValue tn' :: Float)
            putStrLn $ "  False Positive: " ++ show (asValue fp' :: Float)
            putStrLn $ "  False Negative: " ++ show (asValue fn' :: Float)
            putStrLn $ "  Accuracy: " ++ show (asValue acc :: Float)
            putStrLn $ "  Precision: " ++ show (asValue prec :: Float)
            putStrLn $ "  Recall: " ++ show (asValue recall :: Float)
            putStrLn $ "  F1 Score: " ++ show (asValue f1 :: Float)
        ) (zip [1..] cmPairs)

  let accuracy = evalAccuracy tp tn fp fn
      precision = evalPrecision tp fp 
      recall = evalRecall tp fn
      f1Score = calcF1 precision recall
      macroF1 = evalMacroF1 cmPairs
      support = [tp + fn, fp + tn]
      weightedF1 = evalWeightedF1 cmPairs support
      microF1 = evalMicroF1 cmPairs

  putStrLn "\nOverall Evaluation Results:"
  putStrLn $ "Macro F1 Score: " ++ show (asValue macroF1 :: Float)
  putStrLn $ "Weighted F1 Score: " ++ show (asValue weightedF1 :: Float)
  putStrLn $ "Micro F1 Score: " ++ show (asValue microF1 :: Float)

main :: IO ()
main = do
  putStrLn "Reading training CSV file..."
  (inputs, targets) <- loadAdmissionData "Session5/data/chanceOfAdmit_train.csv"
  
  -- initialize model
  initModel <- sample $ MLPSpec
    { feature_counts = [7, 64, 32, 16, 1],
      nonlinearitySpec = Torch.tanh
    }

  -- Training
  putStrLn "Training MLP..."
  trainedModel <- trainMLP initModel inputs targets
  putStrLn "Training completed."

  -- Load evaluation data
  putStrLn "\nReading evaluation CSV file..."
  (evalInputs, evalTargets) <- loadAdmissionData "Session5/data/chanceOfAdmit_eval.csv"

  -- Evaluate model
  putStrLn "Evaluating model..."
  evaluateModel trainedModel evalInputs evalTargets
