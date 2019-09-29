const tf = require('@tensorflow/tfjs-node')
const csv = require('csvtojson')

csv({
  delimiter: ';'
})
  .fromFile('./cotacao-do-bitcoin.csv')
  .then(data => {
    const xData = data.map((_, index, array) => {
      if (index === (array.length - 1)) {
        return [
          // '31.12.2019',
          3709.4,
          3815.1,
          3819.6,
          3658.8
        ]
      }

      const c = array[index + 1]
      return [
        // c.Data,
        Number(c.Fechamento),
        Number(c.Abertura),
        Number(c['Máxima']), 
        Number(c['Mínima'])
      ]
    })

    const yData = data.map(c => [
      // c.Data,
      Number(c.Fechamento),
      Number(c.Abertura),
      Number(c['Máxima']),
      Number(c['Mínima'])
    ])

    const model = tf.sequential()
    const inputLayer = tf.layers.dense({units: 4, inputShape: [4]})

    model.add(inputLayer)

    const learningRate = 0.0000000003
    const optimizer = tf.train.sgd(learningRate)

    model.compile({
      loss: 'meanSquaredError',
      optimizer
    })

    const x = tf.tensor(xData, [xData.length, 4])
    const y = tf.tensor(yData, [yData.length, 4])

    model.fit(x, y, {
      epochs: 1000
    })
      .then(() => {
        const arrInput = [[6984.8, 7190.0, 7518.9, 6802.6]]
        const inputTensor = tf.tensor(arrInput, [1, 4])

        const output = model.predict(inputTensor)

        output.print()
      })
  })