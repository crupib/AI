#!/bin/sh
curl localhost:11434/api/generate -d '{
   "model":"llama3.2",
   "prompt":"Why are trees green?",
   "stream":false
}'
