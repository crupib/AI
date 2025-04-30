#!/bin/sh
curl localhost:11434/api/chat -d '{
   "model":"llama3.2",
   "messages":[{
   "role":"user",
   "content":"Why are trees green. Respond using JSON"
   }],
   "stream":false,
   "format":"json"
}'
