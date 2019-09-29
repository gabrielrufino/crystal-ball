const tf = require('@tensorflow/tfjs-node')

module.exports = (inputsNumber, outputsNumber, loss = 'meanSquaredError', optimizer = 'sgd') => {
  const model = tf.sequential()

  const inputLayer = tf.layers.dense({units: outputsNumber, inputShape: [inputsNumber]}) 
  model.add(inputLayer)

  model.compile({
    loss,
    optimizer
  })

  return model
}
