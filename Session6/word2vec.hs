{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)
import Control.Monad (when, foldM)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', nllLoss', meanDim, matmul, transpose2D, softmax, logSoftmax, binaryCrossEntropyLoss', Dim (..), KeepDim (..))
import Torch.NN (Parameterized (..), Parameter, Linear (..), LinearSpec (..), sample)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.TensorFactories (eye', zeros')
import Torch.Optim (Optimizer, GD (..), runStep, foldLoop)
import Torch.DType (DType(..))

import ML.Exp.Chart (drawLearningCurve)

-- File paths
textFilePath, modelPath, wordLstPath :: FilePath
textFilePath = "Session6/data/review-texts_150.txt"
modelPath =  "Session6/data/embedding_150.params"
wordLstPath = "Session6/data/wordlst_150.txt" 

-- Hyperparameters
learningRate :: Float
learningRate = 0.1
numIters :: Int
numIters = 10

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


-- Probably you should include model and Embedding in the same data class.
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  } deriving (Generic, Parameterized)

data Model = Model {
    mlp :: MLP
  } deriving (Generic, Parameterized)

isUnncessaryChar :: 
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault 0 wrd (M.fromList (zip wordlst [0..length wordlst - 1]))

toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim -- initialize with identity matrix

makeBatches :: Int -> [[Int]] -> [([Int], Int)]
makeBatches n = concatMap (slidingContexts n)

slidingContexts :: Int -> [Int] -> [([Int], Int)]
slidingContexts n xs
  | length xs < 2 * n + 1 = []
  | otherwise = [(context i, xs !! i) | i <- [n .. length xs - n - 1]]
  where
    context i = take n (drop (i - n) xs) ++ take n (drop (i + 1) xs)

main :: IO ()
main = do
  texts <- B.readFile textFilePath
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
      indexedData = map (map wordToIndex) wordLines

  putStrLn $ "Total words: " ++ show (length wordlst)
  putStrLn $ "Indexed data example: " ++ show (take 5 indexedData)
  putStrLn $ "Max index: " ++ show (maximum $ concat indexedData)
  putStrLn $ "Word list size: " ++ show (length wordlst)

  let batchedData = makeBatches 1 indexedData
  -- print batchedData

  let embeddingSpec = EmbeddingSpec { wordNum = length wordlst, wordDim = 9 }
  wordEmb <- makeIndependent $ toyEmbedding embeddingSpec
  let model = Embedding { wordEmbedding = wordEmb }

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  print idxes

  let embTxt = embedding' (toDependent $ wordEmbedding model) (asTensor idxes)
  print $ "embTxt shape: "
  print embTxt -- [the number of sentences, the number of words in each sentence, the embedding dimension]

  --let (contexts, targets) = unzip batchedData
      --contextsTensor = asTensor (contexts :: [[Int]])
      --targetsTensor = asTensor (targets :: [Int])

  -- Training loop
  (trainedEmb, losses) <- foldLoop (wordEmb, []) numIters $ \(state, losses) i -> do
    
    putStrLn $ "Iteration " ++ show i

    -- Process each batch separately within an epoch
    (newState, epochLosses) <- foldLoop (state, []) ((length batchedData) - 1) $ \(batchState, batchLosses) j -> do -- need to fix this later :()
      -- Get single batch
      let (context, target) = batchedData !! j
          contextTensor = asTensor ([context] :: [[Int]])
          targetTensor = asTensor ([target] :: [Int])
      --print $ "contextTensor shape: " ++ show (shape contextTensor)
      --print $ "targetTensor shape: " ++ show (shape targetTensor)
      --print $ "context: " ++ show context
      --print $ "target: " ++ show target
      
      -- Forward pass for this batch
      let contextVecs = embedding' (toDependent batchState) contextTensor
      --print $ "contextVecs shape: " ++ show (shape contextVecs)
      let avgVecs = meanDim (Dim 1) RemoveDim Float contextVecs
      --print $ "avgVecs shape: " ++ show (shape avgVecs)
      --print $ "avgVecs: " ++ show (asValue avgVecs :: [[Float]])
      let scores = matmul avgVecs (transpose2D (toDependent batchState))
      --print $ "scores shape: " ++ show (shape scores)
      let logProbs = logSoftmax (Dim 1) scores
      --print $ "logProbs shape: " ++ show (shape logProbs)
      --print $ "targetsTensor: " ++ show targetTensor
      let loss = nllLoss' targetTensor logProbs 
      let lossVal = asValue loss :: Float
      --print $ "Loss: " ++ show lossVal
      --print $ "Loss shape: " ++ show (shape loss)
      (updatedState, _) <- runStep batchState optimizer loss (asTensor learningRate)
      
      when (j `mod` 300 == 0) $ do
        putStrLn $ "  Batch " ++ show j ++ "/" ++ show (length batchedData) ++ ", Loss: " ++ show lossVal
      
      return (updatedState, lossVal : batchLosses)
      
    -- Calculate average loss for this epoch
    let avgEpochLoss = sum (take (length batchedData) epochLosses) / fromIntegral (length batchedData)
    putStrLn $ "Epoch " ++ show i ++ " completed. Average loss: " ++ show avgEpochLoss
    
    -- return (newState, avgEpochLoss : losses)
    return (newState, losses ++ [avgEpochLoss])

  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

  -- draw Learning Curve
  drawLearningCurve "Session6/charts/word2vec_150_LearningCurve.png" "Learning Curve" [("Training Loss", losses)]

  return ()
  where
    optimizer = GD
