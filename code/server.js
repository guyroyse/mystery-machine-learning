const fs = require('fs').promises
const Redis = require('ioredis')
const express = require('express')
const cors = require('cors')

const PORT = 3000

const TF_MODEL_PATH = './model/mystery-machine-learning.pb'
const TF_MODEL_KEY = 'mystery:tf'
const TF_MODEL_BACKEND = 'TF'
const TF_MODEL_INPUT_NODES = ['x']
const TF_MODEL_OUTPUT_NODES = ['Identity']

const ONNX_MODEL_PATH = './model/mystery-machine-learning.onnx'
const ONNX_MODEL_KEY = 'mystery:onnx'
const ONNX_MODEL_BACKEND = 'ONNX'

const MODEL_DEVICE = 'CPU'

const MAX_LINE_LENGTH = 150

const INPUT_TENSOR_KEY = 'mystery:in'
const INPUT_TENSOR_TYPE = 'FLOAT'
const INPUT_TENSOR_SHAPE = [1, MAX_LINE_LENGTH]

const OUTPUT_TENSOR_KEY = 'mystery:out'

const WORD_INDEX_PATH = './encoders/word_index.json'
const CLASSES_PATH = './encoders/classes.json'

async function main() {

  // connect to redis
  let redis = new Redis()

  // read and load the model
  console.log("Setting the models in RedisAI...")
  
  // read the models from the file system
  let onnxModelBlob = await fs.readFile(ONNX_MODEL_PATH)
  let tfModelBlob = await fs.readFile(TF_MODEL_PATH)
  
  // place the ONNX model into redis
  console.log(`  ONNX: ${ONNX_MODEL_PATH}`)
  let onnxResult = await redis.call(
    'AI.MODELSTORE', ONNX_MODEL_KEY, ONNX_MODEL_BACKEND, MODEL_DEVICE,
    'BLOB', onnxModelBlob)
    
  console.log(`  AI.MODELSTORE result: ${onnxResult}`)

  // place the TensorFlow model into redis
  console.log(`  TensorFlow: ${TF_MODEL_PATH}`)
  let tfResult = await redis.call(
    'AI.MODELSTORE', TF_MODEL_KEY, TF_MODEL_BACKEND, MODEL_DEVICE,
    'INPUTS', TF_MODEL_INPUT_NODES.length, ...TF_MODEL_INPUT_NODES,
    'OUTPUTS', TF_MODEL_OUTPUT_NODES.length, ...TF_MODEL_OUTPUT_NODES,
    'BLOB', tfModelBlob)
  
  console.log(`  AI.MODELSTORE result: ${tfResult}`)

  // load the word index that maps words to numbers
  let wordIndexJson = await fs.readFile(WORD_INDEX_PATH)
  let wordIndex = JSON.parse(wordIndexJson)

  // load the classes for decoding the output
  let classesJson = await fs.readFile(CLASSES_PATH)
  let classes = JSON.parse(classesJson)

  // the request handler for express
  async function handleRequest(req, res) {

    // get the line from the query string
    let backend = (req.body.backend || 'onnx').toLocaleLowerCase()
    let line = req.body.line || ""

    // error if backend is invalid
    if (backend !== 'onnx' && backend !== 'tf') throw "Backend must either ONNX or TF"

    // encode the line
    let encodedLine = line
      .toLowerCase()                    // lower-case only
      .replace(/[^a-z0-9 ]/g, "")       // letters and numbers only
      .replace(/\s+/g, " ")             // single character whitespace
      .trim()                           // no whitespace on the edges
      .split(' ')                       // split the words
      .map(word => wordIndex[word])     // look up the index of the words
      .map(index => index ? index : 0)  // replace words that are not found with 0

    // get the padding needed to bring it up to MAX_LINE_LENGTH
    let paddingLength = MAX_LINE_LENGTH - encodedLine.length
    let padding = new Array(paddingLength).fill().map(_ => 0)

    // concat the paddings and the words to make a full line
    let paddedAndEncodedLine = padding.concat(encodedLine)

    // set the input tensor
    await redis.call(
      'AI.TENSORSET', INPUT_TENSOR_KEY, INPUT_TENSOR_TYPE, ...INPUT_TENSOR_SHAPE,
      'VALUES', ...paddedAndEncodedLine)

    // run the model for ONNX if needed
    if (backend === 'onnx') {
      await redis.call(
        'AI.MODELEXECUTE', ONNX_MODEL_KEY,
        'INPUTS', 1, INPUT_TENSOR_KEY,
        'OUTPUTS', 1, OUTPUT_TENSOR_KEY)  
    }
  
    // run the model for TF if needed
    if (backend === 'tf') {
      await redis.call(
        'AI.MODELEXECUTE', TF_MODEL_KEY,
        'INPUTS', 1, INPUT_TENSOR_KEY,
        'OUTPUTS', 1, OUTPUT_TENSOR_KEY)  
    }
  
    // read the output tensor
    let values = await redis.call('AI.TENSORGET', OUTPUT_TENSOR_KEY, 'VALUES')

    // decode the results
    let results = values
      .map((score, index) => ({ class: index, decodedClass: classes[index], score }))
      .sort((a, b) => b.score - a.score)

    // select the winner
    let winner = results[0]

    console.table(results)

    // send the respsonse
    res.json({ winner, results, line, encodedLine, backend })
  }

  // set up express
  console.log("Setting up express...")
  let app = express()
  app.use(express.json())
  app.use(cors())

  // set up routes
  console.log("Setting up routes...")
  app.post('/jinkies', handleRequest)

  // start express
  console.log("Starting server...")
  app.listen(PORT, () => console.log(`  Listening on port ${PORT}`))
}

main()
