const tf = require('@tensorflow/tfjs-node')
const model = require('../model')(1, 1)

const x = tf.tensor([1, 2, 3, 4], [4, 1])
const y = tf.tensor([10, 20, 30, 40], [4, 1])

model.fit(x, y, {
  epochs: 600
})
  .then(() => {
    const input = tf.tensor([5, 6, 7, 11], [4, 1])
    const output = model.predict(input).flatten().round()
    output.print()
  })
