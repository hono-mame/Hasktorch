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
import Data.Word ()
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (sqrt, embedding', meanDim, Dim (..), KeepDim (..), dot)
import Torch.NN (Parameterized (..), Parameter, Linear (..))
import Torch.Serialize (loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (zeros')
import Torch.DType (DType(..))

-- File paths
modelPath, wordLstPath :: FilePath
modelPath =  "Session6/data/embedding_2000.params"
wordLstPath = "Session6/data/wordlst_2000.txt" 

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

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault 0 wrd (M.fromList (zip wordlst [0..length wordlst - 1]))

sentenceToVec :: Tensor -> [Int] -> Tensor
sentenceToVec emb idxs =
  let idxTensor = asTensor idxs
      vectors = embedding' emb idxTensor
  in meanDim (Dim 0) RemoveDim Float vectors

norm :: Tensor -> Tensor
norm v = Torch.Functional.sqrt (v `dot` v)

cosineSim :: Tensor -> Tensor -> Float
cosineSim v1 v2 =
  let sim = (v1 `dot` v2) / (norm v1 * norm v2)
  in asValue sim

toIndices :: (B.ByteString -> Int) -> String -> [Int]
toIndices wordToIndex s = map wordToIndex $ map (B.pack . encode) (words s)


main :: IO ()
main = do
  -- load the word list
  wordListBytes <- B.readFile wordLstPath
  let wordList = B.split (head $ encode "\n") wordListBytes
      wordToIndex = wordToIndexFactory wordList

  -- load the params that are already trained
  initEmb <- makeIndependent $ zeros' [length wordList, 9]
  Embedding loadedEmb <- loadParams (Embedding initEmb) modelPath
  let embTensor = toDependent loadedEmb

  -- test sentences
  let sentence1 = "I am not sure this is the right site for the question."
  let sentence2 = "I am not sure this question would have made much sense to the Romans themselves."
  putStrLn $ "Sentence 1: " ++ sentence1
  putStrLn $ "Sentence 2: " ++ sentence2

  let vec1 = sentenceToVec embTensor (toIndices wordToIndex sentence1)
  let vec2 = sentenceToVec embTensor (toIndices wordToIndex sentence2)

  putStrLn $ "Vector representation of sentence 1: " ++ show vec1
  putStrLn $ "Vector representation of sentence 2: " ++ show vec2

  let similarity = cosineSim vec1 vec2
  putStrLn $ "Cosine similarity between the two sentences: " ++ show similarity