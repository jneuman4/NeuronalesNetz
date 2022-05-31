class Network (inputTotal: Int, hiddenTotal: Int){

    private val LEARNING_CONSTANT = 0.5

    private val input = mutableListOf<InputNeuron>()
    private val hidden = mutableListOf<HiddenNeuron>()
    private val output = OutputNeuron()

    init {
        //Create input Layer
        for (i in 0 until  inputTotal){
            input.add(InputNeuron())
        }
        input.add(InputNeuron(1.0, true))
        //Create hidden Layer
        for (i in 0 until  hiddenTotal){
            hidden.add(HiddenNeuron())
        }
        hidden.add(HiddenNeuron(1.0, true))

        //Connections
        //Input -> Hidden
        for (i in 0 until input.size){
            for (j in 0 until hidden.size){
                val con = NeuronConnection(input[i], hidden[j])
                input[i].connections.add(con)
                hidden[j].connections.add(con)
            }
        }
        //Hidden -> Output
        for (i in 0 until hidden.size){
            val con = NeuronConnection(input[i], output)
            input[i].connections.add(con)
            output.connections.add(con)
        }
    }

    fun feedForward(inputValues: ArrayList<Double>): Double{//TODO: ArrayList??

        for (i in 0 until inputValues.size){
            input[i].inputVal(inputValues[i])
        }
        for (i in 0 until hidden.size){
            hidden[i].calculateOutput()
        }
        return output.calculateOutput()
    }


    fun train(inputValues: ArrayList<Double>, answer:Double): Double{
        val result = feedForward(inputValues)

        // This is where the error correction all starts
        // Derivative of sigmoid output function * diff between known and guess
        val deltaOutput = result * (1-result) * (answer-result)

        // BACKPROPAGATION
        // This is easier b/c we just have one output
        // Apply Delta to connections between hidden and output
        var connections = output.connections
        for (i in 0 until connections.size){
            val deltaWeight = connections[i].from.output * deltaOutput
            connections[i].adjustWeight(LEARNING_CONSTANT * deltaWeight)
        }

        // ADJUST HIDDEN WEIGHTS
        for (i in 0 until hidden.size){
            connections = hidden[i].connections
            var sum = 0.0

            // Sum output delta * hidden layer connections (just one output)
            for (j in 0 until connections.size){
                // Is this a connection from hidden layer to next layer (output)?
                if (connections[j].from == hidden[i]){
                    sum += connections[j].weight*deltaOutput
                }
            }

            for (j in 0 until connections.size){
                if (connections[j].to == hidden[i]){
                    var deltaHidden = hidden[i].output * (1-hidden[i].output) // Derivate of Sigmoid
                    deltaHidden *= sum
                    val deltaWeight = connections[j].from.output * deltaHidden
                    connections[j].adjustWeight(LEARNING_CONSTANT * deltaWeight)
                }
            }

        }

        return result
    }



}