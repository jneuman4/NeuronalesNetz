import kotlin.random.Random

class NeuronConnection (val from: Neuron, val to: Neuron) {
    var weight: Double = Random.nextDouble()*2-1
}
