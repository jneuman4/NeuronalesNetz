abstract class Neuron(var output: Double = 0.0, val bias: Boolean = false) {

    var derivative = 0.0
    var error = 0.0
    //Sigmoid
    companion object fun f(x: Double) = 1.0/ (1.0 + kotlin.math.exp(-x))

    //All Connections this Neuron is Connected to
    val connections = mutableListOf<NeuronConnection>()

    fun calculateOutput(){
        if (bias) return

        var sum = 0.0
        for (connection in connections){
            // Is this connection moving forward to us
            // Ignore connections that we send our output to
            if (connection.to == this) {
                sum += connection.from.output*connection.weight
            }
        }
        // Output of this Neuron is the result of the sigmoid function
        output = f(sum)
        derivative = output * (1 - output)
    }

}

class InputNeuron(output: Double = 0.0, bias: Boolean = false): Neuron(output, bias) {
    fun inputVal(d: Double){
        output = d
    }
}
class HiddenNeuron(output: Double = 0.0, bias: Boolean = false): Neuron(output, bias)
class OutputNeuron: Neuron()