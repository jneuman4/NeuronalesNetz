class Network (inputTotal: Int, hiddenLayer: IntArray, outputTotal: Int){

    private val LEARNING_CONSTANT = 0.3

    private val inputNeurons = mutableListOf<InputNeuron>()
    private val layers = mutableListOf<Layer>()
    private val outputNeurons = mutableListOf<OutputNeuron>()

    init {
        //Create input Layer
        for (i in 0 until  inputTotal){
            inputNeurons.add(InputNeuron())
        }
        inputNeurons.add(InputNeuron(1.0, true))
        //Create hidden Layer
        for (i in hiddenLayer.indices){
            layers.add(Layer())
            for (j in 0 until hiddenLayer[i]){
                layers[i].hiddenNeurons.add(HiddenNeuron())
            }
            layers[i].hiddenNeurons.add(HiddenNeuron(1.0, true))
        }

        //Create output Layer
        for (i in 0 until  outputTotal){
            outputNeurons.add(OutputNeuron())
        }


        //Connections
        //Input <-> Hidden
        for (i in 0 until inputNeurons.size){
            for (j in 0 until layers[0].hiddenNeurons.size){
                val con = NeuronConnection(inputNeurons[i], layers[0].hiddenNeurons[j])
                inputNeurons[i].connections.add(con)
                layers[0].hiddenNeurons[j].connections.add(con)
            }
        }
        //Hidden <-> Hidden
        if (layers.size > 1){
            for (l in 0 until layers.size-1){
                for (i in 0 until layers[l].hiddenNeurons.size){
                    for (j in 0 until layers[l+1].hiddenNeurons.size) {
                        val con = NeuronConnection(layers[l].hiddenNeurons[i], layers[l+1].hiddenNeurons[j])
                        layers[l].hiddenNeurons[i].connections.add(con)
                        layers[l+1].hiddenNeurons[j].connections.add(con)
                    }
                }
            }
        }

        //Hidden <-> Output
        val lastLayer = layers.last()
        for (i in 0 until lastLayer.hiddenNeurons.size){
            for (j in 0 until outputNeurons.size) {
                val con = NeuronConnection(lastLayer.hiddenNeurons[i], outputNeurons[j])
                lastLayer.hiddenNeurons[i].connections.add(con)
                outputNeurons[j].connections.add(con)
            }
        }
        println("Inputs: "+ inputNeurons.size)
        for (i in hiddenLayer){
            println("Hidden: $i")
        }
        println("Outputs: "+ outputNeurons.size)
    }

    fun feedForward(inputValues: MutableList<Double>){

        for (i in 0 until inputValues.size){
            inputNeurons[i].inputVal(inputValues[i])
        }
        for (l in layers){
            for (hidden in l.hiddenNeurons){
                hidden.calculateOutput()
            }
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
        for (li in layers.lastIndex downTo 0){
            val l = layers[li]
            for (i in l.hiddenNeurons.lastIndex downTo 0){
                val hiddenN = l.hiddenNeurons[i]
                var sum = 0.0
                for (connection in hiddenN.connections){
                    if (hiddenN == connection.from){
                        sum += connection.weight * connection.to.error
                    }
                }
                hiddenN.error = sum * hiddenN.derivative
            }
        }


        //update weights
        for (l in layers){
            for (hiddenN in l.hiddenNeurons){
                val delta = - LEARNING_CONSTANT * hiddenN.error
                for (connection in hiddenN.connections){
                    if (hiddenN == connection.to){
                        connection.weight += delta * connection.from.output
                    }
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
        /*for (outputN in outputNeurons){
            println("Out: " + outputN.output)
            for (connection in outputN.connections){
                if (outputN == connection.to){
                    println("We: " + connection.weight)
                }
            }
        }*/

    }


    fun indexOfHighestValue(): Int{
        var highest = 0
        for (i in 1 until outputNeurons.size){
            if (outputNeurons[i].output > outputNeurons[highest].output) highest = i
        }
        return highest
    }
}