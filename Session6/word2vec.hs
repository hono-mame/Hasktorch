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
import Control.Monad (when)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', nllLoss', meanDim ,matmul, transpose2D, softmax)
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye', zeros')
import Torch.Optim (Optimizer, GD(..), runStep, foldLoop)

import Torch.NN (Linear(..), sample, LinearSpec(..))

-- your text data (try small data first)
textFilePath = "Session6/data/review-texts_150.txt"
modelPath =  "Session6/data/embedding_150.params"
wordLstPath = "Session6/data/wordlst_150.txt" 


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
    context i = left ++ right
      where
        left  = take n $ drop (i - n) xs
        right = take n $ drop (i + 1) xs

learningRate :: Float
learningRate = 0.1


main :: IO ()
main = do
  texts <- B.readFile textFilePath
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
      indexedData = map (map wordToIndex) wordLines
  print wordlst
  
  putStrLn $ "Total words: " ++ show (length wordlst)
  putStrLn $ "Indexed data example: " ++ show (take 5 indexedData)
  putStrLn $ "Max index: " ++ show (maximum $ concat indexedData)
  putStrLn $ "Word list size: " ++ show (length wordlst)

  let batchedData = makeBatches 1 indexedData

  print batchedData

  let embeddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 3}
  wordEmb <- makeIndependent $ toyEmbedding embeddingSpec
  let model = Embedding { wordEmbedding = wordEmb }

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  print idxes
  -- convert to embedding
  let embTxt = embedding' (toDependent $ wordEmbedding model) (asTensor idxes)
  print embTxt

  -- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.

  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  -- saveParams trainedEmb modelPath
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- Load params
  -- initWordEmb <- makeIndependent $ zeros' [1]
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}
  -- loadedEmb <- loadParams initEmb modelPath

  return ()

