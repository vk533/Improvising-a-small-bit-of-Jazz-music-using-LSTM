# Improvising a small bit of Jazz music using LSTM
## Music Generation
Preprocessing of the musical data has been taken care of to render it in terms of musical "values." You can informally think of each "value" as a note, which comprises a pitch and a duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time.

### Loading the Data
X: This is an (m,  Tx , 78) dimensional array.

- We have m training examples, each of which is a snippet of  Tx=30  musical values.
- At each time step, the input is one of 78 different possible values, represented as a one-hot vector.
- For example, X[i,t,:] is a one-hot vector representing the value of the i-th example at time t.

Y: a  (Ty,m,78)  dimensional array

- This is essentially the same as X, but shifted one step to the left (to the past).
- Notice that the data in Y is reordered to be dimension  (Ty,m,78) , where  Ty=Tx . This format makes it more convenient to feed into the LSTM later.
- We're using the previous values to predict the next value. So our sequence model will try to predict  y⟨t⟩  given  x⟨1⟩,…,x⟨t⟩ .

` n_values `: The number of unique values in this dataset. This should be 78.

` indices_values `: python dictionary mapping integers 0 through 77 to musical values.

## Model Overview
- X=(x⟨1⟩,x⟨2⟩,⋯,x⟨Tx⟩)  is a window of size Tx scanned over the musical corpus.
- Each x⟨t⟩ is an index corresponding to a value.
- y^t is the prediction for the next value.

![Image](ImagesJAZZ/Picture1.png)

We will be training the model on random snippets of 30 values taken from a much longer piece of music. Thus, we won't bother to set the first input x⟨1⟩=0⃗ , since most of these snippets of audio start somewhere in the middle of a piece of music. We are setting each of the snippets to have the same length Tx=30 to make vectorization easier.

## Building the model
Now its time to build and train a model that will learn musical patterns. To do so, we will need to build a model that takes in X of shape $(m, T_x, 78)$ and Y of shape $(T_y, m, 78)$. We will use an LSTM with 64 dimensional hidden states. Lets set n_a = 64.

If you're building an RNN where even at test time entire input sequence $x<1>, x<2>, ...... , x^<t> were given in advance, for example if the inputs were words and the output was a label, then Keras has simple built-in functions to build the model. However, for sequence generation, at test time we don't know all the values of x<t> in advance; instead we generate them one at a time using $x<t> = y<t-1>. So the code will be a bit more complicated, and you'll need to implement your own for-loop to iterate over the different time steps.

The function djmodel() will call the LSTM layer t_x times using a for-loop, and it is important that all t_x copies have the same weights. I.e., it should not re-initiaiize the weights every time---the t_x steps should have shared weights. The key steps for implementing layers with shareable weights in Keras are:

- Define the layer objects (we will use global variables for this).
- Call these objects when propagating the input.

## Generating Music
We now have a trained model which has learned the patterns of the jazz soloist. Lets now use this model to synthesize new music.
### Predicting and Sampling
At each step of sampling, we will take as input the activation a and cell state c from the previous state of the LSTM, forward propagate by one step, and get a new output activation as well as cell state. The new activation a can then be used to generate the output, using densor as before.

To start off the model, we will initialize x0 as well as the LSTM activation and and cell value a0 and c0 to be zeros.
