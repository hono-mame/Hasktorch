{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where

import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.List (nub)
import Control.Monad (when)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', nllLoss', meanDim, matmul, transpose2D, logSoftmax, Dim (..), KeepDim (..))
import Torch.NN (Parameterized (..), Parameter)
import Torch.Serialize (saveParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye')
import Torch.Optim (GD (..), runStep, foldLoop)
import Torch.DType (DType(..))

import ML.Exp.Chart (drawLearningCurve)

-- File paths
textFilePath, modelPath, wordLstPath :: FilePath
textFilePath = "Session6/data/review-texts_5000.txt"
modelPath =  "Session6/data/embedding_5000.params"
wordLstPath = "Session6/data/wordlst_5000.txt"

-- Hyperparameters
learningRate :: Float
learningRate = 0.5
numIters :: Int
numIters = 300
batchSize :: Int
batchSize = 50

-- Data types
data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int,
  wordDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst = \wrd -> M.findWithDefault 0 wrd (M.fromList (zip wordlst [0..length wordlst - 1]))

toyEmbedding :: EmbeddingSpec -> Tensor
toyEmbedding EmbeddingSpec{..} = eye' wordNum wordDim

makeBatches :: Int -> [[Int]] -> [([Int], Int)]
makeBatches n = concatMap (slidingContexts n)

slidingContexts :: Int -> [Int] -> [([Int], Int)]
slidingContexts n xs
  | length xs < 2 * n + 1 = []
  | otherwise = [(context i, xs !! i) | i <- [n .. length xs - n - 1]]
  where
    context i = take n (drop (i - n) xs) ++ take n (drop (i + 1) xs)

build :: ((a -> [a] -> [a]) -> [a] -> [a]) -> [a]
build g = g (:) []

chunk :: Int -> [e] -> [[e]]
chunk i ls = map (take i) (build (splitter ls))
 where
  splitter :: [e] -> ([e] -> a -> a) -> a -> a
  splitter [] _ n = n
  splitter l c n = l `c` splitter (drop i l) c n

main :: IO ()
main = do
  putStrLn "Loading data..."
  texts <- B.readFile textFilePath
  putStrLn "checkpoint 1"
  let wordLines = preprocess texts
  putStrLn "checkpoint 2"
  let wordlst = nub $ concat wordLines
  putStrLn "checkpoint 3"
  let wordToIndex = wordToIndexFactory wordlst
  putStrLn "checkpoint 4"
  let indexedData = map (map wordToIndex) wordLines

  --putStrLn $ "Total words: " ++ show (length wordlst)
  --putStrLn $ "Indexed data example: " ++ show (take 5 indexedData)

  let allBatches = makeBatches 1 indexedData
      batchedData = chunk batchSize allBatches

  let embeddingSpec = EmbeddingSpec { wordNum = length wordlst, wordDim = 9 }
  wordEmb <- makeIndependent $ toyEmbedding embeddingSpec
  let model = Embedding { wordEmbedding = wordEmb }

  -- Training loop
  (trainedEmb, losses) <- foldLoop (wordEmb, []) numIters $ \(state, losses) i -> do
    (newState, epochLosses) <- foldLoop (state, []) ((length batchedData)- 1) $ \(batchState, batchLosses) j -> do
      let batch = batchedData !! j
          (contexts, targets) = unzip batch
          contextTensor = asTensor (contexts :: [[Int]])
          targetTensor = asTensor (targets :: [Int])
          contextVecs = embedding' (toDependent batchState) contextTensor
          avgVecs = meanDim (Dim 1) RemoveDim Float contextVecs
          scores = matmul avgVecs (transpose2D (toDependent batchState))
          logProbs = logSoftmax (Dim 1) scores
          loss = nllLoss' targetTensor logProbs
          lossVal = asValue loss :: Float
      (updatedState, _) <- runStep batchState optimizer loss (asTensor learningRate)
      return (updatedState, lossVal : batchLosses)

    let avgEpochLoss = sum epochLosses / fromIntegral (length epochLosses)
    putStrLn $ "Epoch " ++ show i ++ ", Average loss: " ++ show avgEpochLoss
    return (newState, losses ++ [avgEpochLoss])

  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  drawLearningCurve "Session6/charts/word2vec_mini_5000_itr300_LearningCurve.png" "Learning Curve" [("Training Loss", losses)]
  saveParams trainedEmb modelPath
  return ()
  where
    optimizer = GD
