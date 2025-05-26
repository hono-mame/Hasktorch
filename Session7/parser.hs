{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

-- docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/hasktorch-nlp-introduction && stack run day6-parse"

module Main (main) where
-- json
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics

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

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "Session7/data/train.jsonl"

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
  case amazonReviews of
    Left err -> do
      putStrLn $ "Error decoding JSON: " ++ err
      return ()
    Right reviews -> do
      putStrLn $ "Successfully decoded " ++ show (length reviews) ++ " reviews."
      print reviews