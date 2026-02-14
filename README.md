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

As mentioned above, we first fine-tuned gpt-4.1-nano and gpt-4.1-mini models on fold_a and fold_b seperately, and evaluated their performance on the other half. Using small batch sizes (the default when batch_size=auto in the OpenAI fine-tuning API) led to very erratic loss functions. These two models were trained with batch sizes of 4.

We found that the nano model's training loss curves were very erratic. They did not smoothly go down, and the performance of our models trained on the folds, tested accross all datasets was bad.

We then moved to gpt-4.1-mini as the base model for the final submission. The 'final' split (799 train / 90 val) was used to verify that training converged properly and to tune hyperparameters before committing to the full training run. On the held-out 90-user validation set, the fine-tuned mini model achieved **100% accuracy** (19/19 bots caught, 0 false positives), giving us confidence the approach generalizes.

The production model was trained on the 'all' dataset (all 889 labeled examples, no validation holdout) so that it could learn from every available signal before being evaluated on the unseen competition data.

### Prompt design

Each training example is a 3-message conversation in OpenAI's chat format: a system message, a user message, and an assistant message containing just `BOT` or `HUMAN`.

The **system prompt** instructs the model to act as a bot detection expert and lists the signals it should consider: posting patterns, content quality, profile completeness, language patterns, topic diversity, political content, anger/hate-baiting, and unrealistic posting schedules. It also encodes the competition's asymmetric scoring weights (+4 for a correct bot flag, -2 for a wrongly flagged human, -1 for a missed bot) so the model can internalize the cost tradeoffs during training. The prompt ends with "Respond with ONLY `BOT` or `HUMAN` — nothing else."

The **user prompt** is a structured plaintext representation of a single social media account:

```
User ID: <id>
Username: <username>
Name: <name>
Description: <description or (none)>
Location: <location or (none)>
Tweet count: <count>
Z-score (posting activity deviation from average): <z_score>

Posts:
[<timestamp>] [id:<post_id>] [lang:<lang>] <text>
[<timestamp>] [id:<post_id>] [lang:<lang>] <text>
...
```

Posts are sorted chronologically and **no truncation** is applied — the model sees the user's full post history. The z-score is a pre-computed measure of how much a user's posting frequency deviates from the dataset average, giving the model an explicit statistical signal alongside the raw posts.

### Pair-aware cross-validation

The four labeled datasets come in correlated English/French pairs: datasets 30 (EN) + 32 (EN) are paired with 31 (FR) + 33 (FR). Bot campaigns often span both languages in a pair, so a naïve random train/test split would leak information.

To avoid this, we use **pair-aware holdout**: Fold A trains on `30+32` and tests on `31+33`; Fold B trains on `31+33` and tests on `30+32`. A 10% stratified validation split (stratified by label × language) is carved out of each fold's training data for early stopping. This gives us an honest estimate of how the model generalizes to entirely unseen dataset pairs — which mirrors the competition setting where the evaluation datasets are new.

### Inference

At inference time, the JS runtime (`js/detectors/v5.ts` / `js/detectors/final.ts`) reconstructs the same system + user prompt from the raw dataset JSON and sends a single API call per user to the fine-tuned model with `temperature=0` and `max_tokens=8`. The response is trimmed, uppercased, and classified as a bot if it starts with "BOT", otherwise human. Requests are dispatched with a concurrency of 20 and results are cached incrementally to a JSON file, making runs resumable if interrupted.

### Submission

The final detector (`js/detectors/final.ts`) writes a submission file named `maxilillian.detections.<lang>.txt` containing one bot user ID per line. Separate runs are made for the English and French evaluation datasets. The detector also writes a timestamped run file to `runs/` in a format compatible with the JS analysis pipeline for post-hoc accuracy checks.
