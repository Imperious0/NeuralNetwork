#include <iostream>
#include <math.h>

using namespace std;

class NeuralNetwork
{
public:
    int inputCount;
    int neuronLayerCount;
    int* neuronCount;
    int outputCount;

    double (*activation)(double, bool) = tanh;


    double* inputs;
    double** inputsW;
    double*** inputsWError;

    double** netHiddens;
    double** outHiddens;

    double*** hiddensW;
    double*** hiddensWError;

    double** bias;
    double** biasError;

    double* outputs;
    double* outputW;
    double* derOutputs;

    double* biasout;
    double* biasoutError;

    double* target;
    double* errors;


    NeuralNetwork(int inputCount, int hiddenLayerCount, int* hiddenNeuronCount, int outputCount, double* targets)
    {
        this->inputCount = inputCount;
        this->neuronLayerCount = hiddenLayerCount;
        this->neuronCount = hiddenNeuronCount;
        this->outputCount = outputCount;

        inputs = new double[inputCount];
        inputsW = new double* [inputCount];

        inputsWError = new double**[outputCount];
        for (int i = 0; i < outputCount; ++i) {
            inputsWError[i] = new double* [inputCount];
            for (int j = 0; j < inputCount; ++j) {
                inputsWError[i][j] = new double[hiddenNeuronCount[0]];
            }
        }

        for (int i = 0; i < inputCount; ++i) {
            inputsW[i] = new double[hiddenNeuronCount[0]];
        }

        netHiddens = new double* [hiddenLayerCount];
        outHiddens = new double* [hiddenLayerCount];

        hiddensW = new double** [hiddenLayerCount];
        for (int i = 0; i < this->neuronLayerCount; ++i) {

            hiddensW[i] = new double* [neuronCount[i]];
            for (int j = 0; j < neuronCount[i]; ++j) {
                if(i+1 == this->neuronLayerCount)
                {
                    hiddensW[i][j] = new double [this->outputCount];
                }else
                {
                    hiddensW[i][j] = new double [this->neuronCount[i+1]];
                }
            }
        }
        /*hiddensWError = new double*** [outputCount];
        for (int i = 0; i < outputCount; ++i) {*/
        hiddensWError = new double** [this->neuronLayerCount];
        for (int j = 0; j < this->neuronLayerCount; ++j) {
            hiddensWError[j] = new double* [this->neuronCount[j]];
            for (int k = 0; k < this->neuronCount[j]; ++k) {
                if(j+1 == this->neuronLayerCount)
                {
                    hiddensWError[j][k] = new double [this->outputCount];
                }else
                {
                    hiddensWError[j][k] = new double [this->neuronCount[j+1]];
                }
            }
        }
        /*}*/
        bias = new double* [hiddenLayerCount];
        biasError = new double* [hiddenLayerCount];
        for (int i = 0; i < hiddenLayerCount; ++i) {
            netHiddens[i] = new double[hiddenNeuronCount[i]];
            outHiddens[i] = new double[hiddenNeuronCount[i]];

            bias[i] = new double[hiddenNeuronCount[i]];
            biasError[i] = new double[hiddenNeuronCount[i]];
        }

        outputs = new double[outputCount];
        derOutputs = new double[outputCount];
        errors = new double[outputCount];
        biasout = new double[outputCount];
        biasoutError = new double[outputCount];
        target = new double[outputCount];
        for (int i = 0; i < this->outputCount; ++i) {
            target[i] = target[i];
        }


    }

    void initialize()
    {
        cout << "Initializing Started.." << endl;
        for (int i = 0; i < this->inputCount; ++i) {
            for (int j = 0; j < this->neuronCount[0]; ++j) {
                inputsW[i][j] = randomize();
                cout << "InputsW[" << i << "][" << j << "] :" << inputsW[i][j] << endl;
            }
        }
        for (int i = 0; i < this->neuronLayerCount; ++i) {
            for (int j = 0; j < neuronCount[i]; ++j) {
                if(i+1 == this->neuronLayerCount)
                {
                    for (int k = 0; k < this->outputCount; ++k) {
                        hiddensW[i][j][k] = randomize();
                        cout << "HiddensW[" << i << "][" << j << "][" << k << "] :" << hiddensW[i][j][k] << endl;
                    }
                }else
                {
                    for (int k = 0; k < this->neuronCount[i+1]; ++k) {
                        hiddensW[i][j][k] = randomize();
                        cout << "HiddensW[" << i << "][" << j << "][" << k << "] :" << hiddensW[i][j][k] << endl;
                    }
                }
            }
        }
        for (int i = 0; i < this->neuronLayerCount; ++i) {
            for (int j = 0; j < this->neuronCount[i]; ++j) {
                bias[i][j] = randomize();
                cout << "bias[" << i << "][" << j << "] :" << bias[i][j] << endl;
            }
        }
        for (int i = 0; i < this->outputCount; ++i) {
            biasout[i] = randomize();
            cout << "biasout[" << i << "] :" << biasout[i] << endl;
        }
        cout << endl;
        cout << "Initializing Ended.." << endl;
    }


    static double sigmoid(double h, bool derivative)
    {
        if(derivative)
        {
            return h*(1-h);
        }else
        {
            return double(1/(1+exp(-h)));
        }
    }

    static double tanh(double h, bool derivative)
    {
        if(derivative)
        {
            return 1-(pow(h,2.0));
        }else
        {
            return double(1/(1+exp(-h)));
        }
    }

    void another(bool derivative)
    {
        if(derivative)
        {
            for (int i = 0; i < this->outputCount; ++i) {
                this->derOutputs[i] = activation(this->outputs[i], true);
            }
        }else
        {
            for (int i = 0; i < this->outputCount; ++i) {
                this->outputs[i] = activation(this->outputs[i], false);
            }
        }
    }
    void softmax(bool derivative)
    {
        if(derivative)
        {
            double sum_exp = 0.0;
            for (int i = 0; i < this->outputCount; ++i) {
                this->derOutputs[i] = (this->outputs[i]*(1 - this->outputs[i]));
            }
        }else
        {
            double sum_exp = 0.0;

            for (int i = 0; i < this->outputCount; ++i) {
                sum_exp += exp(this->outputs[i]);
            }
            for (int i = 0; i < outputCount; ++i) {
                this->outputs[i] = exp(this->outputs[i]) / (sum_exp);
            }
        }

    }

    double errorCalc(int situation, double out, bool derivative)
    {
        if(derivative)
        {
            return -(target[situation]-out);
        }else
        {
            return (0.5)*(pow((target[situation] - out), 2.0));
        }
    }

    void frontPropagation(double* inputs)
    {
        cout << "Frontpropagation Started.. " << endl;
        for (int i = 0; i < this->neuronLayerCount; ++i) {
            for (int j = 0; j < this->neuronCount[i]; ++j) {

                double sumNet = 0.0;
                if(i > 0)
                {
                    for (int k = 0; k < this->neuronCount[i - 1]; ++k) {
                        sumNet += this->outHiddens[i-1][k] * this->hiddensW[i-1][k][j];
                    }
                }else
                {
                    for (int k = 0; k < this->inputCount; ++k) {
                        sumNet += inputs[k] * inputsW[k][j];
                    }
                }

                sumNet += bias[i][j];
                this->netHiddens[i][j] = sumNet;
                this->outHiddens[i][j] = this->activation(this->netHiddens[i][j], false);

                cout << "netHiddens[" << i << "][" << j << "] :" << netHiddens[i][j] << endl;
                cout << "outHiddens[" << i << "][" << j << "] :" << outHiddens[i][j] << endl;

            }
        }

        for (int i = 0; i < this->outputCount; ++i) {
            double sum_out = 0.0;
            for (int j = 0; j < this->neuronCount[this->neuronLayerCount-1]; ++j) {
                sum_out += this->outHiddens[this->neuronLayerCount-1][j] * this->hiddensW[this->neuronLayerCount-1][j][i];
            }
            sum_out += biasout[i];
            this->outputs[i] = sum_out;
            cout << "outputs[" << i << "] :" << this->outputs[i] << endl;
        }
        if(this->outputCount > 1)
        {
            this->softmax(false);
        }else
            {
                this->another(false);
            }
        for (int i = 0; i < this->outputCount; ++i) {
            cout << "Out Of outputs[" << i << "] :" << this->outputs[i] << endl;
            this->errors[i] = errorCalc(i, this->outputs[i], false);
        }


        cout << endl;
        cout << "Frontpropagation Ended.." << endl;


    }
    void backPropagation()
    {

        if(this->outputCount > 1)
        {
            this->softmax(true);
        }else
            {
                this->another(true);
            }
        //BiasOutErrors
        for (int i = 0; i < outputCount; ++i) {
            biasoutError[i] =  errorCalc(i, this->outputs[i], true) * activation(outputs[i], true);
        }
        //HiddenWErrors
        double sum_errors = 0.0;
        double sum = 1.0;

        for (int i = 0; i < this->outputCount; ++i) {
            for (int j = 0; j < this->neuronCount[this->neuronLayerCount - 1]; ++j) {
                double result = (errorCalc(i, outputs[i], true) * this->derOutputs[i]) * this->hiddensW[this->neuronLayerCount-1][j][i];

                //TODO derOutputs[i] == 0 when target Count 1
                cout << "Searching for, output: " << i << " to Weight: " << j << " on Layer: " << neuronLayerCount-1 << " result is: " << result << " errorCalc(i, outputs[i], true): " << errorCalc(i, outputs[i], true) << " * derOutputs[i] :" << this->derOutputs[i] << endl;
                findOutW(this->neuronCount[this->neuronLayerCount-1], this->neuronLayerCount-1, j, i, result);
            }
        }

        for (int i = 0; i < this->neuronLayerCount; ++i) {
            for (int j = 0; j < this->neuronCount[i]; ++j) {
                if(i+1 == this->neuronLayerCount)
                {
                    for (int k = 0; k < this->outputCount; ++k) {
                        cout << "hiddensWError[" << i << "][" << j << "][" << k << "] :" << hiddensWError[i][j][k] << endl;
                    };
                }else
                    {
                        for (int k = 0; k < this->neuronCount[i+1]; ++k) {
                            cout << "hiddensWError[" << i << "][" << j << "][" << k << "] :" << hiddensWError[i][j][k] << endl;
                        }
                    }

            }
        }
    }

    double findOutW(int outputVal, int neuronLayerVal, int neuronVal, int selectorNeuronVal, double heap)
    {
        if(neuronLayerVal < 1)
        {
            double result = this->outHiddens[neuronLayerVal][neuronVal];
            cout << " result: " << result << " heap:" << heap << endl;
            result *= heap;
            this->hiddensWError[neuronLayerVal][neuronVal][selectorNeuronVal] += result;
            cout << "Layer : " << neuronLayerVal << " i :" << selectorNeuronVal << " neuronVal:" << neuronVal << endl;
        }else
            {
                for (int j = 0; j < this->neuronCount[neuronLayerVal-1]; ++j) {
                    double result = activation(this->outHiddens[neuronLayerVal][neuronVal], true) * this->hiddensW[neuronLayerVal-1][j][neuronVal];
                    result *= heap;
                    cout << "Layer : " << neuronLayerVal << " OutputCount :" << outputVal << " Selected Neuron:" << j << " Selected output :" << selectorNeuronVal << " Next Output Val: " << neuronVal << endl;
                    findOutW(this->neuronCount[neuronLayerVal-1], (neuronLayerVal-1), j, neuronVal, result);
                }
            }

    }
    double randomize()
    {
        return double(rand()/double(RAND_MAX + 1.0));
    }
    void clearMemory()
    {
        delete [] inputs;
        inputs = NULL;

        delete [] inputsW;
        inputsW = NULL;

        delete [] netHiddens;
        netHiddens = NULL;

        delete [] outHiddens;
        outHiddens = NULL;

        delete [] hiddensW;
        hiddensW = NULL;

        delete [] bias;
        bias = NULL;

        delete [] biasout;
        biasout = NULL;

    }
};

int main() {
    int layerCounts[] = {2};
    double inputs[] = {1.0, 1.0};
    double target[] = {0.0, 0.0};
    double *pinputs = inputs;
    NeuralNetwork nn = NeuralNetwork(2, 1, layerCounts, 1, target);
    nn.initialize();
    nn.frontPropagation(pinputs);
    nn.backPropagation();
    //Nnetwork NeuralNetwork = Nnetwork(2,1,layerCounts, 1);


    return 0;
}