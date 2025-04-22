import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, sub, transpose2D)
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

main :: IO ()
main = do
  let sampleA = asTensor ([0.555] :: [Float])
  let sampleB = asTensor ([94.585026] :: [Float])
  let estimatedY = linear (sampleA, sampleB) xs

  -- output
  printOutput(estimatedY)

  let costTensor = cost estimatedY ys
  let costValue = asValue costTensor :: Float
  putStrLn $ "cost is: " ++ show costValue