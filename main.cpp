#include <math.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

const double LR = 0.7;
const double LR_HO = 0.07;
const int tryTime = 1000;

const int situation = 4;
int selected_situation;
const int inputNumber = 2;

const int neuronLayer = 1;
const int neuronNumber = 2;
const int outNumber = 1;

double input[situation][inputNumber];
double inputWeight[situation][inputNumber][neuronNumber];

double neuron[neuronLayer][neuronNumber];
double neuronWeight[neuronLayer][neuronNumber];
double neuronOutWeight[1][neuronNumber][outNumber];
double neuronOut[neuronLayer][neuronNumber];

double bias[neuronLayer + outNumber];

double out[1][outNumber];
double outOutput[outNumber];

double error[outNumber];

double target[situation];

double totalError;

//Functions
double randomize();

double sigmoid(double h);

void calcErrorOfOut();

void setInputs();

void setInputWeights();

void setNeuron();

void setNeuronWeights();

void setNeuronOut();

void setBias();

void setOut();

void setError();

void netHidden(int layer, int layerNeuron);

void outHidden(int layer, int layerNeuron);

void netOutput(int layer, int layerNeuron);

void outOfOutput();

int main()
{
    selected_situation = 3;
    setInputs();
    setInputWeights();
    setBias();
    setNeuron();
    setNeuronWeights();
    setNeuronOut();
    setOut();
    setError();

    for(int i = 0; i < neuronLayer; i++)
    {
        netHidden(i, neuronNumber);
    }

    outHidden(neuronLayer, neuronNumber);
    netOutput(neuronLayer, neuronNumber);
    outOfOutput();
    calcErrorOfOut();

    system("PAUSE");
    return 0;
}

double randomize()
{
    return double(rand() / double(RAND_MAX));
}

void setInputs()
{
    input[0][0] = 0.0;
    input[0][1] = 0.0;
    target[0] = 0.0;

    input[1][0] = 0.0;
    input[1][1] = 1.0;
    target[1] = 1.0;

    input[2][0] = 1.0;
    input[2][1] = 0.0;
    target[2] = 1.0;

    input[3][0] = 1.0;
    input[3][1] = 1.0;
    target[3] = 0.0;

}
void setInputWeights()
{
    for(int i = 0; i < situation; i++)
    {
        for(int j = 0; j < inputNumber; j++)
        {
            for(int k = 0; k < (neuronNumber); k++)
            {
                inputWeight[i][j][k] = randomize();
                cout << (i+1) << ". Situation " << (i*neuronNumber)+j << ". Input Weights:" << inputWeight[i][j][k] << endl;
            }
        }
    }

}
void setBias()
{
    for(int i = 0; i < (neuronLayer+outNumber); i++)
    {
        bias[i] = randomize();
        cout << (i+1) << ". bias :" << bias[i] << endl;
    }
}

void setNeuron()
{
    for(int i = 0; i < neuronLayer; i++)
    {
        for(int j = 0; j < neuronNumber; j++)
        {
            neuron[i][j] = 0.0;
        }
    }
}

void setNeuronWeights()
{
    for(int i = 0; i < neuronLayer; i++)
    {
        for(int j = 0; j < neuronNumber; j++)
        {
            neuronWeight[i][j] = randomize();
            cout << (i+1) << ". Layer " << (i*neuronNumber)+j << ". Neuron Weight:" << neuronWeight[i][j] << endl;
        }
    }
    for (int i = 0; i < neuronNumber; ++i)
    {
        for(int j = 0; j < outNumber; j++)
        {
            neuronOutWeight[0][i][j] = randomize();
            cout << (i+1) << ". Neuron Out Weight: " << neuronOutWeight[0][i][j] << endl;
        }
    }

}

void setNeuronOut()
{
    for(int i = 0; i < neuronLayer; i++)
    {
        for(int j = 0; j < neuronNumber; j++)
        {
            neuronOut[i][j] = randomize();
        }
    }
}

void setOut()
{
    for (int i = 0; i < outNumber; i++)
    {
        out[0][i] = 0.0;
    }
}

void setError()
{
    for (int i = 0; i < outNumber; ++i)
    {
        error[i] = 0.0;
    }
}

void netHidden(int layer, int layerNeuron)
{
    if(layer > 0)
    {
        for(int i = 0; i < layerNeuron; i++)
        {
            for(int j = 0; j < layerNeuron; j++)
            {
                neuron[layer][i] += neuronOut[layer-1][j] * neuronWeight[layer-1][j] + bias[layer];
                cout << layer << ". Layer " << ((i*layerNeuron)+layerNeuron) << ". Net Neuron:" << neuron[layer][i];
            }
        }
    }else
        {
            for(int i = 0; i < inputNumber; i++)
            {
                for(int j = 0; j < layerNeuron; j++)
                {
                    neuron[layer][j] += input[selected_situation][i] * inputWeight[selected_situation][i][j] + bias[layer];
                    cout << layer+1 << ". Layer " << ((i*layerNeuron)+j) << ". Net Neuron: " << neuron[layer][j] << endl;
                }
            }
        }
}

void outHidden(int layer, int layerNeuron)
{
    for(int i = 0; i < layer; i++)
    {
        for(int j = 0; j < layerNeuron; j++)
        {
            neuronOut[i][j] = sigmoid(neuron[i][j]);
            cout << (i+1) << ". Layer " << ((i*layerNeuron)+j) << ". Out Neuron:" << neuronOut[i][j] << endl;
        }
    }
}

void netOutput(int layer, int layerNeuronOut)
{
    for(int i = 0; i < outNumber; i++)
    {
        for(int j = 0; j < layerNeuronOut; j++)
        {
            out[0][i] += neuronOut[layer - 1][j] * neuronOutWeight[0][i][j] + bias[neuronLayer];
            cout << i+1 << ". Net Out: " << out[0][i] << " Neuron Out : " << neuronOut[layer-1][j] << " Neuron Out weight :" << neuronOutWeight[0][i][j] << endl;
        }
    }
}

void outOfOutput()
{
    for(int i = 0; i < outNumber; i++)
    {
        outOutput[i] = sigmoid(out[0][i]);
        cout << i+1 << ". Out of Output: " << outOutput[i] << endl;
    }
}

void calcErrorOfOut()
{
    totalError = 0.0;
    for(int i = 0; i < outNumber; i++)
    {
        totalError += (0.5)*(pow((target[selected_situation] - outOutput[i]), 2.0));
        cout << i+1 << ". Out Of Output Error : " << (0.5)*(pow((target[selected_situation] - outOutput[i]), 2.0)) << " target:" << target[selected_situation] << endl;
    }
    cout << "Total Error: " << totalError << endl;
}

void backPropagation()
{

}
double sigmoid(double h)
{
    return (1/(1+exp(-h)));
}