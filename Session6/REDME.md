# Session6
### TODO
- [x] Watching the video of Bag of words 
- [x] Watching the video of  word2vec
- [ ] Build Bag of words 
- [ ] Build word2vec
- [ ] check my implementation
- [ ] Evaluate the trained model using STS
    - [ ] Prepare data
    - [ ] Evaluate the model (cosine similarity)   

advanced  
- [ ] Calculating the meaning composition (word2vec)
- [ ] Improve the model
- [ ] Make a survey on and implement “negative sampling”.
- [ ]Make a survey on “subword tokenization algorithms”

## Build word2vec

Made smaller text data (review-tests_150.txt).

https://hasktorch.github.io/hasktorch/html/src/Torch.Functional.html#embedding%27
```haskell
embedding' ::
  -- | weights
  Tensor ->
  -- | indices
  Tensor ->
  -- | output
  Tensor
embedding' weights indices =
  unsafePerformIO $
    cast5
      ATen.embedding_ttlbb
      weights
      indices
      (-1 :: Int)
      False
      False
```

```haskell
-- run a single iteration of an optimizer, returning new parameters and updated optimizer state
  runStep :: (Parameterized model) => model -> optimizer -> Loss -> LearningRate -> IO (model, optimizer)
  runStep paramState optState lossValue = runStep' paramState optState (grad' lossValue $ flattenParameters paramState)
```
