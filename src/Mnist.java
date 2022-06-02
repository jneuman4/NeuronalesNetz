import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class Mnist {

    public static void main(String[] args) {
        Network network = new Network(784, 50, 10);
        TrainSet set = createTrainSet(0,4999);
        trainData(network, set);

        TrainSet testSet = createTrainSet(5000,9999);
        testTrainSet(network, testSet);
    }

    public static TrainSet createTrainSet(int start, int end) {

        TrainSet set = new TrainSet(28 * 28, 10);

        try {

            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + "/res/trainImage.idx3-ubyte", "rw");
            MnistLabelFile l = new MnistLabelFile(path + "/res/trainLabel.idx1-ubyte", "rw");

            for(int i = start; i <= end; i++) {
                if(i % 100 ==  0){
                    System.out.println("prepared: " + i);
                }

                List<Double> input = new ArrayList<>();
                List<Double> output = new ArrayList<>();
                for(int j = 0; j < 28*28; j++){
                    input.add((double) m.read() / (double) 256);
                }
                for (int j=0 ; j<10; j++){
                    output.add(0.0);
                }
                output.set(l.readLabel(), 1.0);

                set.addData(input, output);
                m.next();
                l.next();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

         return set;
    }

    public static void trainData(Network net, TrainSet set) {
        for(int i = 0; i < set.size();i++) {
            net.train(set.getInput(i), set.getOutput(i));
            if (i%100 == 0) System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   "+ i + "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
        }
    }

    public static void testTrainSet(Network net, TrainSet set) {
        int correct = 0;
        for(int i = 0; i < set.size(); i++) {
            net.feedForward(set.getInput(i));
            int highest = net.indexOfHighestValue();

            int actualHighest = 0;
            for(int j = 1; j < set.getOutput(i).size(); j++){
                if(set.getOutput(i).get(j) > set.getOutput(i).get(actualHighest)){
                    actualHighest = j;
                }
            }
            if(highest == actualHighest) {

                correct ++ ;
            }
            if(i % 10 == 0) {
                //System.out.println(i + ": " + (double)correct / (double) (i + 1));
                System.out.println(i + ": " +highest +" - " + actualHighest);
            }
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + set.size()+ "  -> " + (double)correct / (double)set.size() +" %");

    }
}
