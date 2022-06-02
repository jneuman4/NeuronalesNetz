import java.util.ArrayList;
import java.util.List;


public class TrainSet {

    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;


    private List<List<Double>> inputData = new ArrayList<>();
    private List<List<Double>> outputData = new ArrayList<>();

    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }

    public void addData(List<Double> in, List<Double> expected) {
        if(in.size() != INPUT_SIZE) return;
        inputData.add(in);
        outputData.add(expected);
    }

    public int size() {
        return inputData.size();
    }

    public List<Double> getInput(int index) {
        if(index >= 0 && index < size())
            return inputData.get(index);
        else return null;
    }

    public List<Double> getOutput(int index) {
        if(index >= 0 && index < size())
            return outputData.get(index);
        else return null;
    }
}
