# Session7
## How to execute the code
```haskell
-- main file for rnn implementation
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session7-rnn"

-- for parsing the input data
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Hasktorch && stack run session7-parser"
```
still trying to fix the error:
```haskell
session7-rnn: Prelude.!!: index too large
```