import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, mul, add, transpose2D)


ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add (mul input slope) intercept

main :: IO ()
main = do
  let sampleA = asTensor ([0.555] :: [Float])
  let sampleB = asTensor ([94.585026] :: [Float])

  let estimatedY = linear (sampleA, sampleB) xs

  let ysList = asValue ys :: [Float]  -- convert Tensor to [Float]
  let estimatedYList = asValue estimatedY :: [Float]  -- convert Tensor to [Float]

  -- output
  mapM_ (\(y, e) -> 
    putStrLn $ "correct answer: " ++ 
    show y ++ "\nestimated: " ++ 
    show e ++ "\n*******")
    (zip ysList estimatedYList)