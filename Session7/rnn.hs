{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Main (main) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import qualified Data.ByteString.Char8 as C8
import Data.List (elemIndex)
import Control.Monad (forM_, foldM)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..), LinearSpec(..), Linear(..), linear)
import Torch.Serialize (loadParams, saveParams)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', softmax, nllLoss', logSoftmax, tanh, add, matmul, Dim(..), meanDim, KeepDim(..))
import Torch.Tensor (Tensor, toDevice, asValue, asTensor,reshape, shape)
import Torch.DType (DType(..))
import Torch.Device (Device(..), DeviceType(..))
import Torch.Optim (GD(..), runStep)

-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

-- Simple RNN layer
data RNNSpec = RNNSpec {
  inputSize :: Int,
  hiddenSize :: Int
} deriving (Show, Eq, Generic)

data RNN = RNN {
  w_ih :: Parameter,  -- input to hidden weights
  w_hh :: Parameter,  -- hidden to hidden weights
  b_ih :: Parameter,  -- input bias
  b_hh :: Parameter   -- hidden bias
} deriving (Show, Generic, Parameterized)

instance Randomizable RNNSpec RNN where
  sample RNNSpec{..} = RNN
    <$> (makeIndependent =<< randnIO' [hiddenSize, inputSize])
    <*> (makeIndependent =<< randnIO' [hiddenSize, hiddenSize])
    <*> (makeIndependent =<< randnIO' [hiddenSize])
    <*> (makeIndependent =<< randnIO' [hiddenSize])

-- RNN forward pass
rnnForward :: RNN -> Tensor -> Tensor -> Tensor
rnnForward RNN{..} input h0 =
  let wx = toDependent w_ih
      wh = toDependent w_hh
      bi = toDependent b_ih
      bh = toDependent b_hh
      -- Simple RNN: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
      -- input should be flattened to 1D vector for matrix multiplication
      flatInput = if length (shape input) > 1 
                  then reshape [-1] input  -- flatten to 1D
                  else input
      step h x = Torch.Functional.tanh (add (add (matmul wx x) bi) (add (matmul wh h) bh))
  in step h0 flatInput

-- model
data ModelSpec = ModelSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int,  -- the dimention of word embeddings
  hiddenDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data Model = Model {
  emb :: Embedding,
  rnn :: RNN,
  output :: Linear
} deriving (Show, Generic, Parameterized)

instance
  Randomizable
    ModelSpec
    Model
  where
    sample ModelSpec {..} = 
      Model
      <$> Embedding <$> (makeIndependent =<< randnIO' [wordDim, wordNum])
      <*> sample (RNNSpec wordDim hiddenDim)
      <*> sample (LinearSpec hiddenDim wordNum)

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {
    emb = loadedEmb,
    rnn = rnn randomizedModel,
    output = output randomizedModel
    }

-- convert text to word indices with bounds checking
textToIndices :: [B.ByteString] -> String -> [Int]
textToIndices wordList text = 
  let words = C8.words (C8.pack text)
      -- Convert C8.ByteString to B.ByteString
      lazyWords = map B.fromStrict words
      maxIndex = length wordList - 1  -- Fix: subtract 1 for 0-based indexing
  in map (\w -> case elemIndex w wordList of
                  Just idx -> if idx <= maxIndex then idx else 0
                  Nothing -> 0) lazyWords


forward :: Model -> [Int] -> Tensor -> Int -> Int -> (Tensor, Tensor)
forward Model{..} indices h0 wordDim hiddenDim =
  let embTensor = embedding' (toDependent $ wordEmbedding emb) (asTensor indices)
      processedEmb = if null indices 
                     then zeros' [wordDim] 
                     else let avgEmb = meanDim (Dim 0) RemoveDim Float embTensor
                          in avgEmb
      finalHidden = rnnForward rnn processedEmb h0
      finalHidden2D = reshape [1, hiddenDim] finalHidden
      outputTensor = linear output finalHidden2D
      logProbs = logSoftmax (Dim 0) outputTensor
  in (logProbs, finalHidden)


train :: Model -> [B.ByteString] -> [AmazonReview] -> Int -> Int -> Int -> IO Model
train model wordList reviews epochs wordDim hiddenDim = do
  let optimizer = GD
      vocabSize = length wordList
  foldM (\currentModel epoch -> do
    putStrLn $ "Epoch " ++ show epoch
    foldM (\m review -> do
      let indices = filter (< vocabSize) $ textToIndices wordList (text review)
      if length indices < 2
        then return m
        else do
          let inputs  = init indices
              targets = tail indices -- inputと1つずらす
              h0 = zeros' [hiddenDim]
              (outputs, _) = forward m inputs h0 wordDim hiddenDim
              targetTensor = asTensor ([head targets] :: [Int])
              --targetTensor = asTensor targets
          putStrLn $ "target Tensor: " ++ show (shape targetTensor)
          putStrLn $ "outputs Tensor: " ++ show (shape outputs)
          let loss = nllLoss' outputs targetTensor
          (updatedModel, _) <- runStep m optimizer loss (asTensor (0.01 :: Float))
          return updatedModel
      ) currentModel reviews
    ) model [1..epochs]

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "Session7/data/train.jsonl"

outputPath :: FilePath
outputPath = "Session7/data/review-texts.txt"

embeddingPath :: FilePath
embeddingPath = "Session6/data/embedding_150.params"

wordLstPath :: FilePath
wordLstPath = "Session6/data/wordlst_150.txt"

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews

  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  putStrLn $ "Loaded " ++ show (length wordLst) ++ " words from vocabulary"

  -- load params (set　wordDim　and wordNum same as session5)
  let modelSpec = ModelSpec {
    wordDim = 9, 
    wordNum = length wordLst,
    hiddenDim = 50
  }
  initModel <- initialize modelSpec embeddingPath

  putStrLn "Starting training..."
  trainedModel <- train initModel wordLst reviews 5 9 50
  
  -- Save trained model
  saveParams trainedModel "Session7/data/trained_rnn.params"
  putStrLn "Training completed and model saved!"
  return ()