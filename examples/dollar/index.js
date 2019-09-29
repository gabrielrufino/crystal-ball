const tf = require('@tensorflow/tfjs-node')
const csv = require('csvtojson')

csv({
  delimiter: ';'
})
  .fromFile('./cotacao-do-dolar.csv')
  .then(data => {
    const xData = data.map((_, index, array) => {
      if (index === (array.length - 1)) {
        return [
          // '31.12.2019',
          3.8813,
          3.8813,
          3.8813,
          3.8813
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

    model.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    })

    const x = tf.tensor(xData, [xData.length, 4])
    const y = tf.tensor(yData, [yData.length, 4])

    model.fit(x, y, {
      epochs: 500
    })
      .then(() => {
        const arrInput = [[3.9285, 3.9708, 3.9781, 3.9251]]
        const inputTensor = tf.tensor(arrInput, [1, 4])

        const output = model.predict(inputTensor)

        output.print()
      })
  })