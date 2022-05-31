abstract class Neuron(var output: Double = 0.0, val bias: Boolean = false) {

    //Sigmoid
    companion object fun f(x: Double) = 1.0/ (1.0 + kotlin.math.exp(-x))

    //All Connections this Neuron is Connected to
    val connections = mutableListOf<NeuronConnection>()

    fun calculateOutput(): Double{
        if (bias) return output

        var sum = 0.0
        for (i in 0 until connections.size){
            // Is this connection moving forward to us
            // Ignore connections that we send our output to
            //TODO: Understand this
            if (connections[i].to == this) {
                sum += connections[i].from.output*connections[i].weight
            }
        }
        // Output of this Neuron is the result of the sigmoid function
        output = f(sum)
        return output
    }

}

class InputNeuron(output: Double = 0.0, bias: Boolean = false): Neuron(output, bias) {
    fun inputVal(d: Double){
        output = d
    }
}
class HiddenNeuron(output: Double = 0.0, bias: Boolean = false): Neuron(output, bias)
class OutputNeuron: Neuron()