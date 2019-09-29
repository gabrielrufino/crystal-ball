const tf = require('@tensorflow/tfjs-node')
const model = require('../model')(3, 1)

const x = tf.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 6], [6, 7, 4]], [5, 3])
const y = tf.tensor([[6], [15], [24], [11], [17]], [5, 1])

model.fit(x, y, {
  epochs: 10000
})
  .then(() => {
    const input = tf.tensor([[10, 20, 30]], [1, 3])
    const output = model.predict(input).dataSync()//.flatten().round()
    const toPrint = output.map(i => Number(i))
    tf.tensor(toPrint).print()
  })


