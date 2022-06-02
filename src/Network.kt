class Network (inputTotal: Int, hiddenTotal: Int, outputTotal: Int){

    private val LEARNING_CONSTANT = 0.3

    private val inputNeurons = mutableListOf<InputNeuron>()
    private val hiddenNeurons = mutableListOf<HiddenNeuron>()
    private val outputNeurons = mutableListOf<OutputNeuron>()

    init {
        //Create input Layer
        for (i in 0 until  inputTotal){
            inputNeurons.add(InputNeuron())
        }
        inputNeurons.add(InputNeuron(1.0, true))
        //Create hidden Layer
        for (i in 0 until  hiddenTotal){
            hiddenNeurons.add(HiddenNeuron())
        }
        hiddenNeurons.add(HiddenNeuron(1.0, true))
        //Create output Layer
        for (i in 0 until  outputTotal){
            outputNeurons.add(OutputNeuron())
        }


        //Connections
        //Input <-> Hidden
        for (i in 0 until inputNeurons.size){
            for (j in 0 until hiddenNeurons.size){
                val con = NeuronConnection(inputNeurons[i], hiddenNeurons[j])
                inputNeurons[i].connections.add(con)
                hiddenNeurons[j].connections.add(con)
            }
        }
        //Hidden <-> Output
        for (i in 0 until hiddenNeurons.size){
            for (j in 0 until outputNeurons.size) {
                val con = NeuronConnection(hiddenNeurons[i], outputNeurons[j])
                hiddenNeurons[i].connections.add(con)
                outputNeurons[j].connections.add(con)
            }
        }
        println("Inputs: "+ inputNeurons.size)
        println("Hidden: "+ hiddenNeurons.size)
        println("Outputs: "+ outputNeurons.size)
    }

    fun feedForward(inputValues: MutableList<Double>){

        for (i in 0 until inputValues.size){
            inputNeurons[i].inputVal(inputValues[i])
        }
        for (i in 0 until hiddenNeurons.size){
            hiddenNeurons[i].calculateOutput()
        }
        for (i in 0 until outputNeurons.size){
            outputNeurons[i].calculateOutput()
        }
    }


    fun train(inputValues: MutableList<Double>, answers:MutableList<Double>){
        //Forwards
        feedForward(inputValues)

        //Backwards
        //calculate errors
        for (i in 0 until outputNeurons.size){
            outputNeurons[i].error = outputNeurons[i].output - answers[i] * outputNeurons[i].derivative
        }
        for (i in hiddenNeurons.lastIndex downTo 0){
            val hiddenN = hiddenNeurons[i]
            var sum = 0.0
            for (connection in hiddenN.connections){
                if (hiddenN == connection.from){
                    sum += connection.weight * connection.to.error
                }
            }
            hiddenN.error = sum * hiddenN.derivative
        }

        //update weights
        for (hiddenN in hiddenNeurons){
            val delta = - LEARNING_CONSTANT * hiddenN.error
            for (connection in hiddenN.connections){
                if (hiddenN == connection.to){
                    connection.weight += delta * connection.from.output
                }
            }
        }
        for (outputN in outputNeurons){
            val delta = - LEARNING_CONSTANT * outputN.error
            for (connection in outputN.connections){
                if (outputN == connection.to){
                    connection.weight += delta * connection.from.output
                }
            }
        }


        //DEBUG
        for (outputN in outputNeurons){
            println("Out: " + outputN.output)
            for (connection in outputN.connections){
                if (outputN == connection.to){
                    println("We: " + connection.weight)
                }
            }
        }

    }


    fun indexOfHighestValue(): Int{
        var highest = 0
        for (i in 1 until outputNeurons.size){
            if (outputNeurons[i].output > outputNeurons[highest].output) highest = i
        }
        return highest
    }
}