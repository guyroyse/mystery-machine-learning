const fs = require('fs').promises
const Redis = require('ioredis')
const express = require('express')

const TF_MODEL_PATH = './model/mystery-machine-learning.pb'
const TF_MODEL_KEY = 'mystery:tf'
const TF_MODEL_BACKEND = 'TF'
const TF_MODEL_INPUT_NODES = ['x']
const TF_MODEL_OUTPUT_NODES = ['Identity']

const ONNX_MODEL_PATH = './model/mystery-machine-learning.onnx'
const ONNX_MODEL_KEY = 'mystery:onnx'
const ONNX_MODEL_BACKEND = 'ONNX'

const MODEL_DEVICE = 'CPU'

const INPUT_TENSOR_KEY = 'mystery:in'
const INPUT_TENSOR_TYPE = 'FLOAT'

const OUTPUT_TENSOR_KEY = 'mystery:out'

const LINE_LENGTH = 500

const PORT = 3000

async function main() {

  // connect to redis
  let redis = new Redis()

  // set up express
  let app = express()
  app.use(express.json())

  // read and load the model
  console.log("Setting the TensorFlow model in RedisAI...")
  console.log(`  Path: ${TF_MODEL_PATH}`)

  // read the model from the file system
  let tfModelBlob = await fs.readFile(TF_MODEL_PATH)

  // place the model into redis
  let tfResult = await redis.call(
    'AI.MODELSET', TF_MODEL_KEY, TF_MODEL_BACKEND, MODEL_DEVICE,
    'INPUTS', ...TF_MODEL_INPUT_NODES,
    'OUTPUTS', ...TF_MODEL_OUTPUT_NODES,
    'BLOB', tfModelBlob)
  
  console.log(`  AI.MODELSET result: ${tfResult}`)  

  // read and load the model
  console.log("Setting the ONNX model in RedisAI...")
  console.log(`  Path: ${ONNX_MODEL_PATH}`)

  // read the model from the file system
  let onnxModelBlob = await fs.readFile(ONNX_MODEL_PATH)

  // place the model into redis
  let onnxResult = await redis.call(
    'AI.MODELSET', ONNX_MODEL_KEY, ONNX_MODEL_BACKEND, MODEL_DEVICE,
    'BLOB', onnxModelBlob)
  
  console.log(`  AI.MODELSET result: ${onnxResult}`)  

  // load the word index that maps words to numbers
  let wordIndexText = await fs.readFile('./encoders/word_index.json')
  let wordIndex = JSON.parse(wordIndexText)

  // load the classes for decoding the output
  let classesText = await fs.readFile('./encoders/classes.json')
  let classes = JSON.parse(classesText)

  // the input tensor shape
  let inputShape = [1, LINE_LENGTH]

  // set up the routes for express
  console.log("Setting up routes...")
  app.all('/jinkies/:line', async (req, res) => {

    // get the line from the query string
    let line = req.params.line || ""

    // encode the line
    let encodedLine = line
      .toLowerCase()                    // lower-case only
      .replace(/[^a-z0-9 ]/g, "")       // letters and numbers only
      .replace(/\s+/g, " ")             // single character whitespace
      .trim()                           // no whitespace on the edges
      .split(' ')                       // split the words
      .map(word => wordIndex[word])     // look up the index of the words
      .map(index => index ? index : 0)  // replace words that are not found with 0

    // get the padding needed to bring it up to LINE_LENGTH
    let paddingLength = LINE_LENGTH - encodedLine.length
    let padding = new Array(paddingLength).fill().map(_ => 0)

    // concat the words and the padding to fully encode the line
    let fullyEncodedLine = encodedLine.concat(padding)

    // set the input tensor
    await redis.call(
      'AI.TENSORSET', INPUT_TENSOR_KEY,
      INPUT_TENSOR_TYPE, ...inputShape,
      'VALUES', ...fullyEncodedLine)

    // run the model
    await redis.call(
      'AI.MODELRUN', ONNX_MODEL_KEY,
      'INPUTS', INPUT_TENSOR_KEY,
      'OUTPUTS', OUTPUT_TENSOR_KEY)
  
    // read the output tensor
    let values = await redis.call('AI.TENSORGET', OUTPUT_TENSOR_KEY, 'VALUES')

    // decode the results
    let results = values
      .map((score, index) => ({ character: classes[index], score }))
      .sort((a, b) => b.score - a.score)

    console.table(results)

    // send the respsonse
    res.send({ line, encodedLine, results })

  })

  // start express
  console.log("Starting server...")
  app.listen(PORT, () => console.log(`  Listening on port ${PORT}`))
}

main()
