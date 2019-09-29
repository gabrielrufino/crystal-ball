const tf = require('@tensorflow/tfjs-node')
const model = require('../model')(3, 2)

const x = tf.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 6], [6, 7, 4]], [5, 3])
const y = tf.tensor([[3, 5], [9, 11], [15, 17], [5, 9], [13, 11]], [5, 2])

model.fit(x, y, {
  epochs: 3000
})
  .then(() => {
    const input = tf.tensor([[10, 20, 30]], [1, 3])
    const output = model.predict(input)//.dataSync()//.flatten().round()
    output.print()
  })


