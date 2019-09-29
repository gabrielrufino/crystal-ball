const tf = require('@tensorflow/tfjs-node')
const model = require('../model')(1, 1)

const x = tf.tensor([1, 2, 3, 4], [4, 1])
const y = tf.tensor([[2, 4], [4, 8], [6, 12], [8, 16]], [4, 2])

model.fit(x, y, {
  epochs: 4000
})
  .then(() => {
    const input = tf.tensor([5, 6, 7, 11], [4, 1])
    const output = model.predict(input).flatten().round()
    output.print()
  })
