# Bot or not challenge 2026

## Project structure
This is a monorepo with js/ and python/ projects. The reason for the js project was mainly to be able to use Vercel's Ai SDK and JS's promise system to speed up the LLM calls. In the python/ project is where we try more customized models and techniques.

The project was setup in a way to be able to iterate quickly (thus the bun cli). There are both ai generated and human written ideas implemented, we just wanted to throw as much as possible and see what would stick.

## V5 detector:

### Preparing datasets

After multiple iterations, we found that these models perform the best using a minimal prompt. The v5 detector runs on fine tuned version of gpt-4.1-mini and gpt-4.1-nano. The reason for using smaller models is that they would be less expensive and they are less prone to 'overthinking'. Having a significantly lower parameter count partly reduces the risk of overfitting and could maybe demonstrate better generalization, even if the number is examples provided are low.

In fine-tune.ipynb, we first create the 4 datasets that would be used for fine tuning. To evaluate whether or not fine-tuning was even a good idea, we first create fold_a and fold_b which are training and validation splits on only half the data. Having fine tuned a model on one half, we test its performance on the other. To our surprise, the accuracy was almost perfect. 

The 'final' dataset is a standard train/val split on the entire set of examples. This was again to verify that our fine-tuning works, and to optimize the hyperparameters of our training before training our model on the entire set of examples. 

Finally, 'all' containt every single example in its training set, with no validation set, and is meant to train the final model that will be used for submission.

### Training

As mentioned above, we first fine-tuned a gpt-4.1-nano model on fold_a and fold_b seperately, and evaluated their performance on the other half. Using small batch sizes (the default when batch_size=auto in the OpenAI fine-tuning API) led to very erratic loss functions. These two models were trained with batch sizes of 4.
